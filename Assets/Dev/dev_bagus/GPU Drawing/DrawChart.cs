using UnityEngine;
using UnityEngine.UI;

public class Chart : MonoBehaviour
{
    public enum ChartType
    {
        Line,
        Bar
    }

    [Header("UI Elements")]
    [SerializeField] ComputeShader _drawComputeShader;
    [SerializeField] RawImage _rawImage;
    [SerializeField] Color _backgroundColour = Color.white;
    [SerializeField] Color _brushColour = Color.black;
    [SerializeField] Color _gridColour = Color.white;
    [Range(1f, 20f)][SerializeField] float _brushSize = 3f;
    [Range(0f, 2f)][SerializeField] float _wiggleSize = 1f;
    [Range(12f, 24f)][SerializeField] int _fontSize = 12;

    [Header("Chart Settings")]
    [SerializeField] private ChartType _chartType = ChartType.Line;
    [SerializeField] private float[] _chartData;
    [SerializeField] private bool _drawAxes = true;
    [SerializeField] private bool _drawGrid = true;
    [SerializeField] private bool _drawLabels = true;
    [SerializeField] private Font _labelFont;
    [SerializeField] private Transform _labelParent;
    [SerializeField] private float _gridSpacing = 100f;
    [SerializeField] private float _paddingLeft = 20f;
    [SerializeField] private float _paddingRight = 20f;
    [SerializeField] private float _paddingTop = 20f;
    [SerializeField] private float _paddingBottom = 20f;
    [SerializeField] private float _labelOffset = 10f;
    
    // Runtime properties 
    RenderTexture _canvasRenderTexture;
    Vector4 _previousMousePos;

    private void Start()
    {
        Init();
    }

    private void Init() {
        Rect rect = _rawImage.rectTransform.rect;
        int rectWidth = Mathf.CeilToInt(rect.width);
        int rectHeight = Mathf.CeilToInt(rect.height);

        _canvasRenderTexture = new RenderTexture(rectWidth, rectHeight, 0, RenderTextureFormat.ARGB32)
        {
            enableRandomWrite = true
        };
        _canvasRenderTexture.Create();

        _rawImage.texture = _canvasRenderTexture;

        int initK = _drawComputeShader.FindKernel("InitBackground");
        _drawComputeShader.SetVector("_BackgroundColour", _backgroundColour);
        _drawComputeShader.SetTexture(initK, "_Canvas", _canvasRenderTexture);
        _drawComputeShader.SetInt("_CanvasWidth", rectWidth);
        _drawComputeShader.SetInt("_CanvasHeight", rectHeight);
        _drawComputeShader.Dispatch(
            initK,
            Mathf.CeilToInt(rectWidth / 8f),
            Mathf.CeilToInt(rectHeight / 8f),
            1
        );
    }
    private void DebugPopulateData() {
        _chartData = new float[52];

        _chartData[0] = 0;
        _chartData[11] = 0;

        for (int i = 1; i < _chartData.Length - 1; i++)
        {
            _chartData[i] = Random.Range(0f, 50f);
        }
    } 

    public void DrawChart()
    {
        // For testing, remove in production
        DebugPopulateData(); 

        if (_chartData == null || _chartData.Length == 0)
        {
            Debug.LogWarning("No chart data available to draw.");
            return;
        }

        Init();

        Rect rect = _rawImage.rectTransform.rect;

        float canvasWidth = _canvasRenderTexture.width;
        float canvasHeight = _canvasRenderTexture.height;
        float chartWidth = canvasWidth - _paddingLeft - _paddingRight;
        float chartHeight = canvasHeight - _paddingTop - _paddingBottom;

        float maxValue = Mathf.Max(_chartData);
        float minValue = Mathf.Min(_chartData);
        int dataCount = _chartData.Length;

        if (_drawAxes)
            DrawAxes();

        if (_drawGrid)
            DrawGrid(minValue, maxValue);

        for (int i = 0; i < dataCount; i++)
        {
            float x = _paddingLeft + i / (float)(dataCount - 1) * chartWidth;
            float y = _paddingBottom + (_chartData[i] - minValue) / (maxValue - minValue) * chartHeight;

            if (_chartType == ChartType.Bar)
            {
                Vector2 start = new Vector2(x, _paddingBottom);
                Vector2 end = new Vector2(x, y);
                DrawLine(start, end, _brushColour, _brushSize, _wiggleSize);
            }
            else if (_chartType == ChartType.Line && i > 0)
            {
                float x0 = _paddingLeft + (i - 1) / (float)(dataCount - 1) * chartWidth;
                float y0 = _paddingBottom + (_chartData[i - 1] - minValue) / (maxValue - minValue) * chartHeight;
                DrawLine(new Vector2(x0, y0), new Vector2(x, y), _brushColour, _brushSize, _wiggleSize);
            }
        }
    }

    private void DrawAxes()
    {
        float left = _paddingLeft;
        float right = _canvasRenderTexture.width - _paddingRight;
        float bottom = _paddingBottom;
        float top = _canvasRenderTexture.height - _paddingTop;

        DrawLine(new Vector2(left, bottom), new Vector2(right, bottom), _gridColour, _brushSize, _wiggleSize);
        DrawLine(new Vector2(left, bottom), new Vector2(bottom, top), _gridColour, _brushSize, _wiggleSize);

        if (_drawLabels)
            CreateLabel("0", new Vector2(0, 0));
    }

    void DrawLine(Vector2 from, Vector2 to, Color brushColor, float brushSize, float wiggleSize)
    {
        int updateKernel = _drawComputeShader.FindKernel("Update");

        _drawComputeShader.SetBool("_MouseDown", true);
        _drawComputeShader.SetVector("_PreviousMousePosition", new Vector4(from.x, from.y, 0, 0));
        _drawComputeShader.SetVector("_MousePosition", new Vector4(to.x, to.y, 0, 0));
        _drawComputeShader.SetFloat("_BrushSize", brushSize);
        _drawComputeShader.SetFloat("_WiggleSize", wiggleSize);
        _drawComputeShader.SetVector("_BrushColour", brushColor);
        _drawComputeShader.SetInt("_CanvasWidth", _canvasRenderTexture.width);
        _drawComputeShader.SetInt("_CanvasHeight", _canvasRenderTexture.height);
        _drawComputeShader.SetTexture(updateKernel, "_Canvas", _canvasRenderTexture);

        _drawComputeShader.Dispatch(
            updateKernel,
            Mathf.CeilToInt(_canvasRenderTexture.width / 8f),
            Mathf.CeilToInt(_canvasRenderTexture.height / 8f),
            1
        );
    }

    private void DrawGrid(float minVal, float maxVal)
    {
        if (_chartData == null || _chartData.Length == 0) return;

        // Clear previous labels
        foreach (Transform child in _labelParent)
            Destroy(child.gameObject);

        float left = _paddingLeft;
        float right = _canvasRenderTexture.width - _paddingRight;
        float bottom = _paddingBottom;
        float top = _canvasRenderTexture.height - _paddingTop;

        float chartWidth = right - left;
        float chartHeight = top - bottom;

        int dataCount = _chartData.Length;

        // === Y-Axis horizontal grid lines and labels ===
        int numYLines = Mathf.FloorToInt(chartHeight / _gridSpacing);
        float valuePerLine = (maxVal - minVal) / numYLines;

        for (int i = 0; i <= numYLines; i++)
        {
            float y = bottom + i * _gridSpacing;
            if (y < bottom || y > top) continue;

            DrawLine(new Vector2(left, y), new Vector2(right, y), _gridColour, 1f, _wiggleSize);

            if (_drawLabels)
            {
                float value = minVal + i * valuePerLine;
                if (y >= bottom && y <= top)
                    CreateLabel(value.ToString("0.#"), new Vector2(left - _labelOffset, y));
            }
        }
        int divider = Mathf.FloorToInt(dataCount / 10f);

        // === X-Axis vertical lines and labels at each data point ===
        for (int i = 0; i < dataCount; i++)
        {
            float x = (_chartType == ChartType.Bar)
                ? left + chartWidth * (i + 0.5f) / dataCount
                : left + chartWidth * i / (dataCount - 1f);

            // Draw vertical grid only for Line chart
            if (_chartType == ChartType.Line && x > left && x < right)
            {
                if (i % divider == 0)
                    DrawLine(new Vector2(x, bottom), new Vector2(x, top), _gridColour, 1f, _wiggleSize);
            }

            // Always show first and last label, skip others if too close to edge
            bool isFirst = i == 0;
            bool isLast = i == dataCount - 1;

            if (_drawLabels && (isFirst || isLast || (x > left + 10f))) //&& x < right - 10f)))
            {
                if (i % divider == 0)
                    CreateLabel($"{i}", new Vector2(x, bottom - _labelOffset));
            }
        }
    }

    private void CreateLabel(string text, Vector2 canvasPos)
    {
        if (_labelFont == null || _labelParent == null) 
            return;

        GameObject labelObj = new GameObject("Label", typeof(RectTransform));
        labelObj.transform.SetParent(_labelParent, false);

        Text label = labelObj.AddComponent<Text>();
        label.text = text;
        label.font = _labelFont;
        label.fontSize = _fontSize;
        label.color = Color.black;
        label.alignment = TextAnchor.MiddleCenter;
        label.horizontalOverflow = HorizontalWrapMode.Overflow;
        label.verticalOverflow = VerticalWrapMode.Overflow;

        RectTransform rt = label.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(60, 20);     
        rt.anchorMin = rt.anchorMax = new Vector2(0, 0);    // anchor to bottom-left
        rt.pivot = new Vector2(0.5f, 0.5f);                 // center it

        //Vector2 offset;
        //if (Mathf.Approximately(canvasPos.y, 0))      // X-axis label → move down
        //    offset = new Vector2(0, -(label.fontSize+_paddingBottom));
        //else if (Mathf.Approximately(canvasPos.x, 0)) // Y-axis label → move left
        //    offset = new Vector2(-(label.fontSize + _paddingLeft), 0);
        //else
        //    offset = Vector2.zero;

        rt.anchoredPosition = canvasPos; //+ offset;
    }
}
