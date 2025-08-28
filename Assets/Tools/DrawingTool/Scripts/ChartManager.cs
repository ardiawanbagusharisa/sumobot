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
    [Range(12f, 24f)][SerializeField] int _sidePanelFontSize = 14;

    [Header("ChartManager Settings")]
    [SerializeField] private bool _enableDebugData = true;
    [SerializeField] private bool _enablePallete = true;
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
    [SerializeField] private int _numSeries = 3;

    ColorPalette _colorPalette;
    RenderTexture _canvasRenderTexture;

    private void DebugPopulateSeries()
    {
        _chartSeriesList.Clear();

        for (int i = 0; i < _numSeries; i++)
        {
            float[] data = new float[52];

            // Add 0 values to first and last index. 
            data[0] = data[data.Length - 1] = 0;

            for (int j = 1; j < data.Length - 1; j++)
                data[j] = Random.Range(0f, 50f);

            _chartSeriesList.Add(ChartSeries.Create($"Series {i + 1}", data, (ChartSeries.ChartType)(i % 2), Random.ColorHSV()));

        }
    }

    private void Start()
    {
        Init();
        if (_enableDebugData)
            DebugPopulateSeries();
        if (_enablePallete)
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
            _chartSeriesList[i].axesColor = generatedColors[i];
        }

        Debug.Log("Drawing tool palette initialized.");
    }

    public void Setup(
        int? xGridSpacing = null,
        float? yGridSpacing = null,
        System.Func<float, string> onXLabelCreated = null)
    {
        if (xGridSpacing != null)
            _xGridDataSpacing = (int)xGridSpacing;
        if (yGridSpacing != null)
            _yGridSpacing = (float)yGridSpacing;
    }

    public void AddChartSeries(ChartSeries chart)
    {
        var idx = _chartSeriesList.FindIndex((x) => x.name == chart.name);
        if (idx == -1)
            _chartSeriesList.Add(chart);
        else
        {
            UpdateChartSeries(chart);
        }
    }
    public void UpdateChartSeries(ChartSeries chart)
    {
        int chartIndex = _chartSeriesList.FindIndex((x) => x.name == chart.name);
        if (chartIndex != -1)
        {
            _chartSeriesList[chartIndex] = chart;
        }
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
            label.color = series.axesColor;
            label.font = _labelFont;
            label.fontSize = _sidePanelFontSize;

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

    public void ClearChartSeries()
    {
        _chartSeriesList.Clear();
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

            // Add zero at first and end index
            if (series.data[0] != 0)
            {
                var temp = series.data.ToList();
                temp.Insert(0, 0);
                series.data = temp.ToArray();
            }
            if (series.data.Last() != 0)
            {
                var temp = series.data.ToList();
                temp.Add(0);
                series.data = temp.ToArray();
            }

            series.OnPrepareToDraw?.Invoke();

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
        {
            DrawLabels(globalMin, globalMax, chartWidth, chartHeight,
                    _chartSeriesList.Where(s => s.isVisible && s.data != null).Select(s => s.data.Length).DefaultIfEmpty(0).Max());

        }


        // Loop through all series
        for (var index = 0; index < _chartSeriesList.Count; index++)
        {
            var series = _chartSeriesList[index];
            if (!series.isVisible || series.data == null || series.data.Length == 0)
                continue;

            if (series.chartType == ChartSeries.ChartType.GroupBar)
            {
                float minVal = series.data.Min();
                float maxVal = series.data.Max();
                float range = Mathf.Max(0.0001f, maxVal - minVal);

                float groupWidth = chartWidth / series.groupCount;
                float barWidth = groupWidth / series.categoryCount;
                for (int group = 0; group < series.groupCount; group++)
                {
                    for (int cat = 0; cat < series.categoryCount; cat++)
                    {
                        int groupIndex = group * series.categoryCount + cat;
                        if (groupIndex >= series.data.Length) continue;

                        float value = series.data[groupIndex];
                        if (value == 0)
                            continue;
                        if (groupIndex < 1 || series.categoryNames.Count() == 0)
                            continue;
                        string catName = series.categoryNames[groupIndex - 1];
                        float y = _paddingBottom + (value - minVal) / range * chartHeight;
                        float x = _paddingLeft + group * groupWidth + cat * barWidth;

                        // Draw rectangle instead of line
                        // Rect barRect = new Rect(x, _paddingBottom, barWidth, y - _paddingBottom);

                        Color barColor = Color.green;
                        if (groupIndex <= Mathf.FloorToInt(series.categoryCount / series.groupCount))
                        {
                            barColor = Color.green;
                        }
                        else
                        {
                            barColor = Color.red;
                        }

                        for (int dx = 0; dx < Mathf.CeilToInt(barWidth); dx++)
                        {
                            float xPos = x + dx;
                            Vector2 start = new Vector2(xPos, _paddingBottom);
                            Vector2 end = new Vector2(xPos, y);

                            DrawLine(start, end, barColor, 1f, 1f);
                        }

                        // === Vertical Label inside the bar ===
                        Vector2 labelPos = new Vector2(
                            x + barWidth / 2f,
                            y + value / 2f // middle of the bar height
                        );

                        CreateLabel(catName, labelPos, Color.white, rotation: 90f);

                    }
                }
            }

            else
             if (series.chartType == ChartSeries.ChartType.Bar)
            {
                // === TIGHT-PACKED GROUPED BAR CHART (NO SPACING) ===
                float[] data = series.data;
                int dataCount = data.Length;

                if (dataCount <= 1) continue; // Skip if no meaningful bars

                // === Step 1: Set group X position (no spacing between products) ===
                // Use only one bar per group, but place both bars at same X (touching)
                float groupX = _paddingLeft + (index * 0f); // 0 spacing — no gap between products
                float barWidth = 1f; // Full width — no spacing

                // === Step 2: Draw each bar (no spacing between them) ===
                for (int i = 0; i < dataCount; i++)
                {
                    float x = groupX + (i * 0f); // No spacing — all bars at same X offset (0 spacing)
                    float y = _paddingBottom + (data[i] - globalMin) / (globalMax - globalMin) * chartHeight;

                    // Draw vertical bar from bottom to y
                    Vector2 start = new Vector2(x, _paddingBottom);
                    Vector2 end = new Vector2(x, y);
                    DrawLine(start, end, series.axesColor, series.brushSize, _wiggleSize);
                }
            }
            else if (series.chartType == ChartSeries.ChartType.Line)
            {
                // === ORIGINAL LINE CHART (no change, just draw as before) ===
                float[] data = series.data;
                int dataCount = data.Length;

                for (int i = 1; i < dataCount; i++)
                {
                    float x0 = _paddingLeft + (i - 1) / (float)(dataCount - 1) * chartWidth;
                    float y0 = _paddingBottom + (data[i - 1] - globalMin) / (globalMax - globalMin) * chartHeight;
                    float x1 = _paddingLeft + (i) / (float)(dataCount - 1) * chartWidth;
                    float y1 = _paddingBottom + (data[i] - globalMin) / (globalMax - globalMin) * chartHeight;

                    DrawLine(
                        new Vector2(x0, y0),
                        new Vector2(x1, y1),
                        series.axesColor,
                        series.brushSize,
                        _wiggleSize
                    );
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

    void DrawLine(Vector2 from, Vector2 to, Color brushColor, float brushSize, float wiggleSize, string id = "default")
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

    private List<(int index, float x)> CalculateXAxisGridPoints(int dataCount, float chartWidth)
    {
        var points = new List<(int index, float x)>();
        for (int i = 0; i < dataCount; i++)
        {
            float x = _paddingLeft + i / (float)(dataCount - 1) * chartWidth;
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
        // Clean up old labels
        foreach (Transform child in _labelParent)
            Destroy(child.gameObject);

        var visibleSeries = _chartSeriesList.Where(s => s.isVisible && s.data != null && s.data.Length > 0).ToList();

        // === Y-axis Labels (per series or global) ===
        var yGridPoints = CalculateYAxisGridPoints(minVal, maxVal, chartHeight);
        foreach (var (value, y) in yGridPoints)
        {
            var series = visibleSeries.FirstOrDefault(s => s.data.Contains(value)); // or any logic
            if (series != null && series.onYLabelCreated != null)
            {
                string label = series.onYLabelCreated(value);
                CreateLabel(label, new Vector2(_paddingLeft - _labelOffset, y));
            }
            else
            {
                CreateLabel(value.ToString("0.#"), new Vector2(_paddingLeft - _labelOffset, y));
            }
        }

        // === X-axis Labels for GroupBar ===
        foreach (var series in visibleSeries)
        {
            if (series.chartType == ChartSeries.ChartType.GroupBar)
            {
                for (int group = 0; group < series.groupCount; group++)
                {
                    float groupWidth = chartWidth / series.groupCount;

                    // Center label under the group
                    float x = _paddingLeft + group * groupWidth + groupWidth / 2f;

                    // Use series formatter if provided
                    string groupLabel = series.onXLabelCreated != null
                        ? series.onXLabelCreated(group).ToString()
                        : $"G{group + 1}";

                    CreateLabel(groupLabel, new Vector2(x, _paddingBottom - _labelOffset), series.labelColor);
                }
            }
            else
            {
                // Fallback: original per-index logic
                int dataCount = series.data.Length;
                if (dataCount == 0) continue;

                var xPoints = CalculateXAxisGridPoints(dataCount, chartWidth);

                foreach (var (index, x) in xPoints)
                {
                    if (index == 0) continue;

                    string xLabel = series.onXLabelCreated != null
                        ? series.onXLabelCreated(index).ToString()
                        : index.ToString();

                    CreateLabel(xLabel, new Vector2(x, _paddingBottom - _labelOffset), series.labelColor);
                }
            }
        }


        // Always show 0 on X-axis (centered)
        CreateLabel("0", new Vector2(_paddingLeft - _labelOffset, _paddingBottom - _labelOffset));
    }



    private void CreateLabel(string text, Vector2 canvasPos, Color? labelColor = null, float rotation = 0f)
    {
        if (_labelFont == null || _labelParent == null)
            return;

        GameObject labelObj = new GameObject("Label", typeof(RectTransform));
        labelObj.transform.SetParent(_labelParent, false);

        Text label = labelObj.AddComponent<Text>();
        label.text = text;
        label.font = _labelFont;
        label.fontSize = _fontSize;
        label.color = labelColor ?? Color.white;
        label.alignment = TextAnchor.MiddleCenter;
        label.horizontalOverflow = HorizontalWrapMode.Overflow;
        label.verticalOverflow = VerticalWrapMode.Overflow;

        RectTransform rt = label.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(100, 30);

        rt.localRotation = Quaternion.Euler(0, 0, rotation);

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
        Bar,
        GroupBar
    }

    public string name;
    public float[] data; // Flattened array: [val1, val2, ..., valN] where N = groupCount * categoryCount
    public Color axesColor;
    public Color labelColor;
    public float brushSize;
    public bool isVisible = true;
    public ChartType chartType;

    public System.Action OnPrepareToDraw;

    // ✅ X-axis label formatter (e.g., "Player 1", "Player 2")
    public System.Func<int, string> onXLabelCreated;

    // ✅ Y-axis label formatter (e.g., "50%", "100")
    public System.Func<float, string> onYLabelCreated;

    // ✅ NEW: Group count (how many groups in the dataset)
    public int groupCount { get; set; }

    // ✅ NEW: category count (how many actions per group)
    public int categoryCount { get; set; }

    // ✅ NEW: group names (e.g., ["Player 1", "Player 2"])
    public string[] groupNames { get; set; }

    // ✅ NEW: category names (e.g., ["Buy", "Chat", "Like"])
    public string[] categoryNames { get; set; }

    // ✅ NEW: per-category colors (optional)
    public Color[] categoryColors { get; set; }

    public static ChartSeries Create(
        string name,
        float[] data,
        ChartType type,
        Color? axesColor = null,
        float brushSize = 3f,
        Color? labelColor = null
    )
    {
        var chart = new ChartSeries();
        chart.name = name;
        chart.data = data;
        chart.chartType = type;
        chart.axesColor = axesColor ?? Color.white;
        chart.brushSize = brushSize;
        chart.labelColor = labelColor ?? Color.white;

        // Set default formatters
        chart.onXLabelCreated = index => index.ToString();
        chart.onYLabelCreated = value => value.ToString("0.0");
        return chart;
    }

    public static ChartSeries CreateGroup(
        string name,
        float[] data,
        int groupCount,
        int categoryCount,
        string[] groupNames,
        string[] categoryNames,
        float brushSize = 3f,
        Color[] categoryColors = null,
        Color? labelColor = null,
        Color? axesColor = null
    )
    {
        var chart = Create(name, data, ChartType.GroupBar, axesColor, brushSize, labelColor);

        chart.groupCount = groupCount;
        chart.categoryCount = categoryCount;
        chart.groupNames = groupNames;
        chart.categoryNames = categoryNames;
        List<Color> catColors = new();
        if (categoryColors == null)
        {
            for (var i = 0; i < groupCount; i++)
            {
                catColors.Add(Color.white);
            }
            chart.categoryColors = catColors.ToArray();
        }
        else
        {
            chart.categoryColors = categoryColors;
        }

        return chart;
    }

}
