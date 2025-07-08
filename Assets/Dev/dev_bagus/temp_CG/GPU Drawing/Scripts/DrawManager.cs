using SumoHelper;
using UnityEngine;
using UnityEngine.UI;

public class DrawManager : MonoBehaviour
{
    [Header("Shader & Colors")]
    [SerializeField] ComputeShader _drawComputeShader;
    [SerializeField] Color _backgroundColour = Color.white;
    [SerializeField] Color _brushColour = Color.black;

    [Header("Brush Settings")]
    [SerializeField] BrushSlider _brushSizeSlider;
    [SerializeField] BrushSlider _wiggleSldier;
    [Range(1f, 20f)][SerializeField] float _brushSize = 4f;
    [Range(0f, 2f)][SerializeField] float _wiggleSize = 1f;
    [SerializeField] bool _useSoftBrush = true;
    [Range(0f, 1f)][SerializeField] float _softnessLevel = 0.5f;

    [Header("UI Elements")]
    [SerializeField] RawImage _rawImage;
    [SerializeField] Button _saveButton;

    RenderTexture _canvasRT;
    Vector4 _previousMousePos;

    // [Edit later] Temporary 
    public enum ChartType { 
        Line, 
        Bar
    }
    [Header("Chart Settings")]
    [SerializeField] private Button _drawChartButton;
    [SerializeField] private ChartType _chartType = ChartType.Line;
    [SerializeField] private float[] _chartData;
    [SerializeField] private bool _drawAxes = true;
    [SerializeField] private bool _drawGridLines = true;
    [SerializeField] private int _gridLineCount = 5;
    [SerializeField] private bool _showLabels = true;
    [SerializeField] private Font _labelFont;
    [SerializeField] private Transform _labelParent;
    [SerializeField] private float _gridSpacing = 100f;
    [SerializeField] private float _paddingLeft = 20f;
    [SerializeField] private float _paddingRight = 20f;
    [SerializeField] private float _paddingTop = 20f;
    [SerializeField] private float _paddingBottom = 20f;
    //

    void Awake()
    {
        // Hook save button
        _saveButton.onClick.AddListener(SaveCurrentCanvas);
        _drawChartButton.onClick.AddListener(DrawChart);

    }

    void Start()
    {
        // 1) Make RT same size as RawImage
        var rect = _rawImage.rectTransform.rect;
        int rw = Mathf.CeilToInt(rect.width);
        int rh = Mathf.CeilToInt(rect.height);

        _canvasRT = new RenderTexture(rw, rh, 0, RenderTextureFormat.ARGB32)
        {
            enableRandomWrite = true
        };
        _canvasRT.Create();

        // 2) Assign RT to RawImage
        _rawImage.texture = _canvasRT;

        // 3) Clear RT via compute shader
        int initK = _drawComputeShader.FindKernel("InitBackground");
        _drawComputeShader.SetVector("_BackgroundColour", _backgroundColour);
        _drawComputeShader.SetTexture(initK, "_Canvas", _canvasRT);
        _drawComputeShader.SetInt("_CanvasWidth", rw);
        _drawComputeShader.SetInt("_CanvasHeight", rh);
        _drawComputeShader.Dispatch(
            initK,
            Mathf.CeilToInt(rw / 8f),
            Mathf.CeilToInt(rh / 8f),
            1
        );

        // 4) Slider hookup
        _brushSizeSlider.slider.SetValueWithoutNotify(_brushSize);
        _brushSizeSlider.slider.onValueChanged.AddListener(sz => _brushSize = sz);
        _wiggleSldier.slider.SetValueWithoutNotify(_wiggleSize);
        _wiggleSldier.slider.onValueChanged.AddListener(wz => _wiggleSize = wz);

        _previousMousePos = Input.mousePosition;
    }

    void Update()
    {
        // Don’t draw when slider is in use or mouse up
        if (_brushSizeSlider.isInUse || _wiggleSldier.isInUse || !Input.GetMouseButton(0) )
        {
            _previousMousePos = Input.mousePosition;
            return;
        }

        // Map screen mouse → RT coords
        RectTransformUtility.ScreenPointToLocalPointInRectangle(
            _rawImage.rectTransform,
            Input.mousePosition,
            null,
            out Vector2 localCurr);

        RectTransformUtility.ScreenPointToLocalPointInRectangle(
            _rawImage.rectTransform,
            _previousMousePos,
            null,
            out Vector2 localPrev);

        var r = _rawImage.rectTransform.rect;
        float cw = _canvasRT.width, ch = _canvasRT.height;
        var curr = new Vector4(
            (localCurr.x - r.x) / r.width * cw,
            (localCurr.y - r.y) / r.height * ch,
            0, 0);
        var prev = new Vector4(
            (localPrev.x - r.x) / r.width * cw,
            (localPrev.y - r.y) / r.height * ch,
            0, 0);

        // Dispatch draw
        int updK = _drawComputeShader.FindKernel("Update");
        _drawComputeShader.SetBool("_MouseDown", true);
        _drawComputeShader.SetVector("_PreviousMousePosition", prev);
        _drawComputeShader.SetVector("_MousePosition", curr);
        _drawComputeShader.SetBool("_UseSoftBrush", _useSoftBrush);
        _drawComputeShader.SetFloat("_BrushSize", _brushSize);
        _drawComputeShader.SetFloat("_SoftnessLevel", _softnessLevel);
        _drawComputeShader.SetFloat("_WiggleSize", _wiggleSize);
        _drawComputeShader.SetVector("_BrushColour", _brushColour);
        _drawComputeShader.SetInt("_CanvasWidth", (int)cw);
        _drawComputeShader.SetInt("_CanvasHeight", (int)ch);
        _drawComputeShader.SetTexture(updK, "_Canvas", _canvasRT);

        _drawComputeShader.Dispatch(updK, Mathf.CeilToInt(cw / 8f),
            Mathf.CeilToInt(ch / 8f),
            1
        );

        _previousMousePos = Input.mousePosition;
    }

    void SaveCurrentCanvas()
    {
        // Build unique file path
        string baseName = "Sprite";
        string directory = Application.dataPath;
        string ext = ".png";
        string path = System.IO.Path.Combine(directory, baseName + ext);
        int counter = 1;
        while (System.IO.File.Exists(path))
        {
            path = System.IO.Path.Combine(directory, $"{baseName}_{counter}{ext}");
            counter++;
        }

        // Capture RT → Texture2D
        var prevRT = RenderTexture.active;
        RenderTexture.active = _canvasRT;

        var tex = new Texture2D(_canvasRT.width, _canvasRT.height, TextureFormat.ARGB32, false);
        tex.ReadPixels(new Rect(0, 0, _canvasRT.width, _canvasRT.height),
                       0, 0);
        tex.Apply();

        // Make background transparent
        Color bg = _backgroundColour;
        Color[] pixels = tex.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            // How “far” is this pixel from pure background?
            float diff = Vector3.Distance(
                new Vector3(pixels[i].r, pixels[i].g, pixels[i].b),
                new Vector3(bg.r, bg.g, bg.b)
            );

            if (diff < 0.05f)       // tweak tolerance as needed
                pixels[i].a = 0f;   // fully transparent
            else
                pixels[i].a = 1f;   // or keep fully opaque
        }
        tex.SetPixels(pixels);
        tex.Apply();

        // Write PNG
        System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
        Debug.Log($"Saved PNG to: {path}");

        // Cleanup
        RenderTexture.active = prevRT;
        Destroy(tex);

#if UNITY_EDITOR
        UnityEditor.AssetDatabase.Refresh();
#endif
    }

    public void SetBrushColor(Color brushColor) {
        _brushColour = brushColor;
    }

    // [Edit later] Temporary 
    public void DrawChart() {
        _chartData = new float[5];
        for (int i = 0; i < _chartData.Length; i++)
        {
            _chartData[i] = Random.Range(0f, 50f);
        }

        if (_chartData == null || _chartData.Length == 0) { 
            Debug.LogWarning("No chart data available to draw.");
            return;
        }

        // Clear the canvas before drawing
        float maxValue = Mathf.Max(_chartData);
        float minValue = Mathf.Min(_chartData);
        Rect rect = _rawImage.rectTransform.rect;
        float cw = _canvasRT.width;
        float ch = _canvasRT.height;
        
        float chartWidth = cw - _paddingLeft - _paddingRight;
        float chartHeight = ch - _paddingTop - _paddingBottom;

        int dataCount = _chartData.Length;

        // Draw Chart Elements 
        if (_drawAxes)
            DrawAxes();

        if (_drawGridLines)
            DrawGrids(minValue, maxValue);

        for (int i = 0; i < dataCount; i++) { 
            float x = _paddingLeft + i / (float)(dataCount - 1) * chartWidth;
            float y = _paddingBottom + (_chartData[i] - minValue) / (maxValue - minValue) * chartHeight;
            //(1 - (_chartData[i] - minValue) / (maxValue - minValue)) * ch;

            if (_chartType == ChartType.Bar) { 
                Vector2 start = new Vector2(x, _paddingBottom);
                Vector2 end = new Vector2(x, y);
                DrawBrushLine(start, end);
            }
            else if (_chartType == ChartType.Line && i > 0)
            {
                float x0 = _paddingLeft + (i - 1) / (float)(dataCount - 1) * chartWidth;
                float y0 = _paddingBottom + (_chartData[i - 1] - minValue) / (maxValue - minValue) * chartHeight;

                DrawBrushLine(new Vector2(x0, y0), new Vector2(x, y));
            }
        }
    }

    void DrawBrushLine(Vector2 from, Vector2 to) { 
        int updK = _drawComputeShader.FindKernel("Update");

        _drawComputeShader.SetBool("_MouseDown", true);
        _drawComputeShader.SetVector("_PreviousMousePosition", new Vector4(from.x, from.y, 0, 0));
        _drawComputeShader.SetVector("_MousePosition", new Vector4(to.x, to.y, 0, 0));
        _drawComputeShader.SetBool("_UseSoftBrush", _useSoftBrush);
        _drawComputeShader.SetFloat("_BrushSize", _brushSize);
        _drawComputeShader.SetFloat("_SoftnessLevel", _softnessLevel);
        _drawComputeShader.SetFloat("_WiggleSize", _wiggleSize);
        _drawComputeShader.SetVector("_BrushColour", _brushColour);
        _drawComputeShader.SetInt("_CanvasWidth", _canvasRT.width);
        _drawComputeShader.SetInt("_CanvasHeight", _canvasRT.height);
        _drawComputeShader.SetTexture(updK, "_Canvas", _canvasRT);

        _drawComputeShader.Dispatch(
            updK,
            Mathf.CeilToInt(_canvasRT.width / 8f),
            Mathf.CeilToInt(_canvasRT.height / 8f),
            1
        );
    }

    void DrawGridLine(Vector2 from, Vector2 to)
    {
        int updK = _drawComputeShader.FindKernel("Update");

        _drawComputeShader.SetBool("_MouseDown", true);
        _drawComputeShader.SetVector("_PreviousMousePosition", new Vector4(from.x, from.y, 0, 0));
        _drawComputeShader.SetVector("_MousePosition", new Vector4(to.x, to.y, 0, 0));
        _drawComputeShader.SetBool("_UseSoftBrush", _useSoftBrush);
        _drawComputeShader.SetFloat("_BrushSize", 1f);
        _drawComputeShader.SetFloat("_SoftnessLevel", _softnessLevel);
        _drawComputeShader.SetFloat("_WiggleSize", _wiggleSize);
        _drawComputeShader.SetVector("_BrushColour", Color.white);
        _drawComputeShader.SetInt("_CanvasWidth", _canvasRT.width);
        _drawComputeShader.SetInt("_CanvasHeight", _canvasRT.height);
        _drawComputeShader.SetTexture(updK, "_Canvas", _canvasRT);

        _drawComputeShader.Dispatch(
            updK,
            Mathf.CeilToInt(_canvasRT.width / 8f),
            Mathf.CeilToInt(_canvasRT.height / 8f),
            1
        );
    }

    private void DrawAxes() {
        float left = _paddingLeft;
        float right = _canvasRT.width - _paddingRight;
        float bottom = _paddingBottom;
        float top = _canvasRT.height - _paddingTop;

        DrawBrushLine(new Vector2(left, bottom), new Vector2(right, bottom));
        DrawBrushLine(new Vector2(left, bottom), new Vector2(bottom, top));

        if (_showLabels)
        {
            CreateLabel("0", new Vector2(0, 0));
        }
    }

    private void DrawGrids(float minVal, float maxVal) {
        if (_gridSpacing <= 0f) return;

        float left = _paddingLeft;
        float right = _canvasRT.width - _paddingRight;
        float bottom = _paddingBottom;
        float top = _canvasRT.height - _paddingTop;

        float chartWidth = right - left;
        float chartHeight = top - bottom;

        if (_drawGridLines)
        {
            _drawComputeShader.SetFloat("_BrushSize", 1f);
            _drawComputeShader.SetVector("_BrushColour", Color.white);

            int lineCountY = Mathf.FloorToInt(chartHeight / _gridSpacing);
            for (int i = 1; i <= lineCountY; i++)
            {
                float y = bottom + i * _gridSpacing;
                DrawGridLine(new Vector2(left, y), new Vector2(right, y));

                if (_showLabels)
                {
                    float value = minVal + (maxVal - minVal) * (y - bottom / chartHeight);
                    CreateLabel(value.ToString("0.##"), new Vector2(left, y));
                }
            }

            int lineCountX = Mathf.FloorToInt(chartWidth / _gridSpacing);
            for (int i = 1; i <= lineCountX; i++)
            {
                float x = left + i * _gridSpacing;
                DrawGridLine(new Vector2(x, bottom), new Vector2(x, top));

                if (_showLabels)
                {
                    int index = Mathf.RoundToInt(x - left / chartWidth * (_chartData.Length - 1));
                    index = Mathf.Clamp(index, 0, _chartData.Length - 1);
                    CreateLabel($"X{index}", new Vector2(x, bottom));
                }
            }
        }
    }

    private void CreateLabel(string text, Vector2 canvasPos) {
        if (_labelFont == null || _labelParent == null) return;

        GameObject labelObj = new GameObject("Label", typeof(RectTransform));
        labelObj.transform.SetParent(_labelParent, false);

        Text label = labelObj.AddComponent<Text>();
        label.text = text;
        label.font = _labelFont;
        label.fontSize = 14;
        label.color = Color.black;
        label.alignment = TextAnchor.MiddleCenter;
        label.horizontalOverflow = HorizontalWrapMode.Overflow;
        label.verticalOverflow = VerticalWrapMode.Overflow;

        RectTransform rt = label.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(60, 20);  // Adjust size as needed
        rt.anchorMin = rt.anchorMax = new Vector2(0, 0); // anchor to bottom-left
        rt.pivot = new Vector2(0.5f, 0.5f); // center it

        // Intelligent offset
        Vector2 offset;
        if (Mathf.Approximately(canvasPos.y, 0))      // X-axis label → move down
            offset = new Vector2(0, -12);
        else if (Mathf.Approximately(canvasPos.x, 0)) // Y-axis label → move left
            offset = new Vector2(-28, 0);
        else
            offset = Vector2.zero;

        rt.anchoredPosition = canvasPos + offset;
    }
    //
}
