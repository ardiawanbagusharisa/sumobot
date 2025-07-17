using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class ChartManager : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField] ComputeShader _drawComputeShader;
    [SerializeField] RawImage _rawImage;
    [SerializeField] Color _backgroundColour = Color.white;
    [SerializeField] Color _gridColour = Color.white;
    [Range(1f, 20f)][SerializeField] float _brushSize = 3f;
    [Range(0f, 2f)][SerializeField] float _wiggleSize = 1f;
    [Range(12f, 24f)][SerializeField] int _fontSize = 12;

    [Header("ChartManager Settings")]
    [SerializeField] private bool _drawAxes = true;
    [SerializeField] private bool _drawGrid = true;
    [SerializeField] private bool _drawLabels = true;
    [SerializeField] private Font _labelFont;
    [SerializeField] private Transform _labelParent;
    [SerializeField] private int _xGridDataSpacing = 10;
    [SerializeField] private float _yGridSpacing = 100f;
    [SerializeField] private float _paddingLeft = 20f;
    [SerializeField] private float _paddingRight = 20f;
    [SerializeField] private float _paddingTop = 20f;
    [SerializeField] private float _paddingBottom = 20f;
    [SerializeField] private float _labelOffset = 10f;

    [Header("SidePanel Settings")]
    [SerializeField] private RectTransform _sidePanelParent;
    [SerializeField] private GameObject _togglePrefab;
    [SerializeField] private List<ChartSeries> _chartSeriesList = new List<ChartSeries>();

    [Header("Chart Colors Source")] 
    [SerializeField] private Texture2D _paletteSourceTexture; 
    [SerializeField] private int _numFixColors = 0;

    ColorPalette _colorPalette;
    RenderTexture _canvasRenderTexture;

    private void DebugPopulateSeries()
    {
        _chartSeriesList.Clear();

        for (int i = 0; i < 3; i++)
        {
            float[] data = new float[52];
            
            // Add 0 values to first and last index. 
            data[0] = data[data.Length - 1] = 0; 

            for (int j = 1; j < data.Length - 1; j++) 
                data[j] = Random.Range(0f, 50f); 

            _chartSeriesList.Add(new ChartSeries($"Series {i + 1}", data, (ChartSeries.ChartType)(i % 2), Random.ColorHSV()));
        }
    }

    private void Start()
    {
        Init();
        DebugPopulateSeries();
        InitPalette();
        InitSidePanel();
        DrawChart();
    }

    private void OnEnable()
    {
        DrawChart();
    }

    private void OnRectTransformDimensionsChange()
    {
        Init();
        DrawChart();
    }

    private void Init()
    {
        Rect rect = _rawImage.rectTransform.rect;
        int rectWidth = Mathf.CeilToInt(rect.width);
        int rectHeight = Mathf.CeilToInt(rect.height);

        if (_canvasRenderTexture == null || _canvasRenderTexture.width != rectWidth || _canvasRenderTexture.height != rectHeight)
        {
            if (_canvasRenderTexture != null)
            {
                _canvasRenderTexture.Release();
                Destroy(_canvasRenderTexture);
            }

            _canvasRenderTexture = new RenderTexture(rectWidth, rectHeight, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
            _canvasRenderTexture.Create();
            _rawImage.texture = _canvasRenderTexture;
        }

        ClearCanvas(_canvasRenderTexture, _backgroundColour);
    }

    public void InitPalette()
    {
        if (_paletteSourceTexture == null || _chartSeriesList.Count == 0)
        {
            Debug.LogError("Source Texture is missing! Cannot generate palette.");
            return;
        }

        _colorPalette = new ColorPalette(_paletteSourceTexture, _numFixColors);
        List<Color> generatedColors = _colorPalette.GenerateColors(_chartSeriesList.Count, false);

        if (generatedColors == null || generatedColors.Count == 0)
        {
            Debug.LogError("Generated colors list is empty! Cannot apply colors to chart series.");
            return;
        }

        for (int i = 0; i < _chartSeriesList.Count; i++)
        {
            _chartSeriesList[i].color = generatedColors[i];
        }

        Debug.Log("Drawing tool palette initialized.");
    }

    public void InitSidePanel()
    {
        if (_sidePanelParent == null || _togglePrefab == null || _chartSeriesList == null)
            return;

        foreach (Transform toggle in _sidePanelParent)
            Destroy(toggle.gameObject);

        foreach (ChartSeries series in _chartSeriesList)
        {
            GameObject toggleObj = Instantiate(_togglePrefab, _sidePanelParent);
            Toggle toggle = toggleObj.GetComponent<Toggle>();
            Text label = toggleObj.GetComponentInChildren<Text>();

            toggle.isOn = series.isVisible;
            label.text = series.name;
            label.color = series.color;
            label.font = _labelFont;

            toggle.onValueChanged.AddListener(isOn =>
            {
                series.isVisible = isOn;
                DrawChart();
            });
        }

        LayoutRebuilder.ForceRebuildLayoutImmediate(_sidePanelParent);
    }
    void ClearCanvas(RenderTexture canvas, Color color)
    {
        int initK = _drawComputeShader.FindKernel("InitBackground");
        _drawComputeShader.SetVector("_BackgroundColour", color);
        _drawComputeShader.SetTexture(initK, "_Canvas", canvas);
        _drawComputeShader.SetInt("_CanvasWidth", canvas.width);
        _drawComputeShader.SetInt("_CanvasHeight", canvas.height);
        _drawComputeShader.Dispatch(initK, Mathf.CeilToInt(canvas.width / 8f), Mathf.CeilToInt(canvas.height / 8f), 1);

        foreach (Transform child in _labelParent)
            Destroy(child.gameObject);
    }

    public void DrawChart()
    {
        ClearCanvas(_canvasRenderTexture, _backgroundColour);

        if (_chartSeriesList == null || _chartSeriesList.Count == 0 || !_chartSeriesList.Any(s => s.isVisible))
            return;

        float canvasWidth = _canvasRenderTexture.width;
        float canvasHeight = _canvasRenderTexture.height;
        float chartWidth = canvasWidth - _paddingLeft - _paddingRight;
        float chartHeight = canvasHeight - _paddingTop - _paddingBottom;

        // Scaling 
        float globalMin = float.MaxValue;
        float globalMax = float.MinValue;
        bool hasVisibleData = false;

        foreach (var series in _chartSeriesList)
        {
            if (!series.isVisible || series.data == null || series.data.Length == 0) 
                continue;

            globalMin = Mathf.Min(globalMin, series.data.Min());
            globalMax = Mathf.Max(globalMax, series.data.Max());
            hasVisibleData = true;
        }

        if (!hasVisibleData) 
            return; 

        if (Mathf.Approximately(globalMax, globalMin))
        {
            globalMin -= 1f;
            globalMax += 1f;
        }

        // Draw grid, axes, label, and chart series
        if (_drawAxes)
            DrawAxes();

        if (_drawGrid)
            DrawGrid(globalMin, globalMax, chartWidth, chartHeight);

        if (_drawLabels) 
            DrawLabels(globalMin, globalMax, chartWidth, chartHeight,
                    _chartSeriesList.Where(s => s.isVisible && s.data != null).Select(s => s.data.Length).DefaultIfEmpty(0).Max());
        
        foreach (ChartSeries series in _chartSeriesList)
        {
            if (!series.isVisible || series.data == null || series.data.Length == 0) 
                continue;

            float[] data = series.data;
            int dataCount = data.Length;

            for (int i = 0; i < dataCount; i++)
            {
                float x = _paddingLeft + i / (float)(dataCount - 1) * chartWidth;
                float y = _paddingBottom + (data[i] - globalMin) / (globalMax - globalMin) * chartHeight;

                if (i == 0 || i == dataCount - 1)
                    continue;

                if (series.chartType == ChartSeries.ChartType.Bar)
                {
                    Vector2 start = new Vector2(x, _paddingBottom);
                    Vector2 end = new Vector2(x, y);
                    DrawLine(start, end, series.color, _brushSize, _wiggleSize);
                }
                else if (series.chartType == ChartSeries.ChartType.Line && i > 1)
                {
                    float x0 = _paddingLeft + (i - 1) / (float)(dataCount - 1) * chartWidth;
                    float y0 = _paddingBottom + (data[i - 1] - globalMin) / (globalMax - globalMin) * chartHeight;
                    DrawLine(new Vector2(x0, y0), new Vector2(x, y), series.color, _brushSize, _wiggleSize);
                }
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
        DrawLine(new Vector2(left, bottom), new Vector2(left, top), _gridColour, _brushSize, _wiggleSize);
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

    private List<(float value, float y)> CalculateYAxisGridPoints(float minVal, float maxVal, float chartHeight)
    {
        var points = new List<(float value, float y)>();
        int desiredNumYLabels = Mathf.Max(1, Mathf.FloorToInt(chartHeight / _yGridSpacing));
        float range = maxVal - minVal;
        float rawStep = range / desiredNumYLabels;

        float p = Mathf.Floor(Mathf.Log10(rawStep));
        float norm = rawStep / Mathf.Pow(10, p);
        float niceN = (norm >= 5) ? 5 : (norm >= 2) ? 2 : 1;
        float actualStep = niceN * Mathf.Pow(10, p);

        float firstVal = Mathf.Ceil(minVal / actualStep) * actualStep;
        for (float val = firstVal; val <= maxVal + actualStep * 0.5f; val += actualStep)
        {
            float y = _paddingBottom + (val - minVal) / range * chartHeight;
            points.Add((val, y));
        }

        return points;
    }

    private List<(int index, float x)> CalculateXAxisGridPoints(int maxLen, float chartWidth)
    {
        var points = new List<(int index, float x)>();
        if (maxLen <= 0)
            return points;

        float left = _paddingLeft;
        int spacing = Mathf.Max(1, _xGridDataSpacing);
        float xSpacingFactor = Mathf.Max(1, maxLen - 1);

        for (int i = 0; i < maxLen; i += spacing)
        {
            float x = left + chartWidth * i / xSpacingFactor;
            points.Add((i, x));
        }

        return points;
    }


    private void DrawGrid(float minVal, float maxVal, float chartWidth, float chartHeight)
    {
        float left = _paddingLeft;
        float right = _canvasRenderTexture.width - _paddingRight;
        float bottom = _paddingBottom;
        float top = _canvasRenderTexture.height - _paddingTop;

        int maxLen = _chartSeriesList.Where(s => s.isVisible && s.data != null).Select(s => s.data.Length).DefaultIfEmpty(0).Max();
        if (maxLen == 0) 
            return;

        var yPoints = CalculateYAxisGridPoints(minVal, maxVal, chartHeight);
        foreach (var (value, y) in yPoints)
        {
            DrawLine(new Vector2(left, y), new Vector2(right, y), _gridColour, 1f, _wiggleSize);
        }

        var xPoints = CalculateXAxisGridPoints(maxLen, chartWidth);
        foreach (var (index, x) in xPoints)
        {
            DrawLine(new Vector2(x, bottom), new Vector2(x, top), _gridColour, 1f, _wiggleSize);
        }
    }

    private void DrawLabels(float minVal, float maxVal, float chartWidth, float chartHeight, int maxLen)
    {
        // clean labels
        foreach (Transform child in _labelParent)
            Destroy(child.gameObject);

        var yPoints = CalculateYAxisGridPoints(minVal, maxVal, chartHeight);
        foreach (var (value, y) in yPoints)
            if (value != 0) CreateLabel(value.ToString("0.#"), new Vector2(_paddingLeft - _labelOffset, y));

        var xPoints = CalculateXAxisGridPoints(maxLen, chartWidth);
        foreach (var (index, x) in xPoints)
            if (index != 0) CreateLabel(index.ToString(), new Vector2(x, _paddingBottom - _labelOffset));

        CreateLabel("0", new Vector2(_paddingLeft - _labelOffset, _paddingBottom - _labelOffset));
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
        rt.sizeDelta = new Vector2(100, 30);

        RectTransform rawImageRectTransform = _rawImage.rectTransform;
        Vector2 rawImageLocalPoint = canvasPos; 
        Vector3 worldPointOfCanvasPos = rawImageRectTransform.TransformPoint(rawImageLocalPoint + rawImageRectTransform.rect.min);
        Vector2 screenPoint = RectTransformUtility.WorldToScreenPoint(null, worldPointOfCanvasPos);

        Vector2 localPoint;
        RectTransform labelParentRectTransform = _labelParent.GetComponent<RectTransform>();
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(labelParentRectTransform, screenPoint, null, out localPoint))
        {
            rt.anchoredPosition = localPoint;
        }
    }
}

[System.Serializable]
public class ChartSeries
{
    public enum ChartType
    {
        Line,
        Bar
    }

    public string name;
    public float[] data;
    public Color color;
    public bool isVisible = true;
    public ChartType chartType;

    public ChartSeries(string name, float[] data, ChartType chartType, Color color)
    {
        this.name = name;
        this.data = data;
        this.chartType = chartType;
        this.color = color;
    }
}