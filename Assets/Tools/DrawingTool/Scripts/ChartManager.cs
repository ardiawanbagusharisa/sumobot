using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class ChartManager : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField] RawImage _rawImage;
    [SerializeField] Color _backgroundColour = Color.white;
    [SerializeField] Color _gridColour = Color.white;
    [Range(1f, 20f)][SerializeField] float _brushSize = 3f;
    [Range(0f, 2f)][SerializeField] float _wiggleSize = 1f;
    [Range(12f, 32f)][SerializeField] int _fontSize = 12;
    [Range(12f, 24f)][SerializeField] int _sidePanelFontSize = 14;

    [Header("ChartManager Settings")]
    [SerializeField] private bool _enableDebugData = true;
    [SerializeField] private bool _enablePallete = true;
    [SerializeField] private bool _drawAxes = true;
    [SerializeField] private bool _drawGrid = true;
    [SerializeField] private bool _drawLabels = true;
    [SerializeField] private Font _labelFont;
    [SerializeField] private Transform _labelParent;
    [SerializeField] public int XGridDataSpacing = 10;
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

    [SerializeField] ComputeShader drawShader;
    [SerializeField] Material drawMaterial;
    ColorPalette _colorPalette;
    public Drawer drawer;

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

            _chartSeriesList.Add(ChartSeries.Create($"Series {i + 1}", ChartSeries.ChartType.Bar, Random.ColorHSV(), data));

        }
    }

    private void Start()
    {
        if (_enableDebugData)
            DebugPopulateSeries();
        if (_enablePallete)
            InitPalette();
        InitSidePanel();
        DrawChart();
    }

    void OnEnable()
    {
        Init();
    }

    // private void OnEnable()
    // {
    //     DrawChart();
    // }

    // private void OnRectTransformDimensionsChange()
    // {
    //     Init();
    //     DrawChart();
    // }

    private void Init()
    {
        drawer = Drawer.CreateDrawer(_rawImage, drawMaterial, drawShader, _backgroundColour);
        drawer.Init();
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

    public void AddChartSeries(ChartSeries chart, bool sidePanel = false)
    {
        var idx = _chartSeriesList.FindIndex((x) => x.name == chart.name);
        if (idx == -1)
        {
            if (sidePanel)
                AddSidePanel(chart);
            _chartSeriesList.Add(chart);
        }
        else
            UpdateChartSeries(chart);
    }

    public ChartSeries GetChartSeries(string name)
    {
        return _chartSeriesList.Find((el) => el.name == name);
    }
    public void UpdateChartSeries(ChartSeries chart)
    {
        int chartIndex = _chartSeriesList.FindIndex((x) => x.name == chart.name);
        if (chartIndex != -1)
        {
            bool isCurrentChartVisible = _chartSeriesList[chartIndex].isVisible;
            _chartSeriesList[chartIndex] = chart;
            _chartSeriesList[chartIndex].isVisible = isCurrentChartVisible;
        }
    }

    public void InitSidePanel()
    {
        if (_sidePanelParent == null || _togglePrefab == null || _chartSeriesList == null)
            return;

        ClearSidePanels();

        foreach (ChartSeries series in _chartSeriesList)
        {
            AddSidePanel(series);
        }

        LayoutRebuilder.ForceRebuildLayoutImmediate(_sidePanelParent);
    }
    private void AddSidePanel(ChartSeries series)
    {
        GameObject toggleObj = Instantiate(_togglePrefab, _sidePanelParent);
        Toggle toggle = toggleObj.GetComponent<Toggle>();
        Text label = toggleObj.GetComponentInChildren<Text>();

        toggle.isOn = series.isVisible;
        label.text = series.name;
        label.color = series.labelColor;
        label.font = _labelFont;
        label.fontSize = _sidePanelFontSize;

        toggle.onValueChanged.AddListener(isOn =>
        {
            ChartSeries runningChart = _chartSeriesList.Find((el) => el.name == series.name);
            if (runningChart != null)
            {
                runningChart.isVisible = isOn;
                DrawChart();
            }

            series.onVisibilityChanged.Invoke(isOn);
        });
    }


    public void ClearChartSeries()
    {
        _chartSeriesList.Clear();
    }

    public void ClearSidePanels()
    {
        foreach (Transform toggle in _sidePanelParent)
            Destroy(toggle.gameObject);
    }

    private void ClearCanvas()
    {
        drawer.ClearCanvas(_backgroundColour);
        foreach (Transform child in _labelParent)
            Destroy(child.gameObject);
    }

    public void DrawChart()
    {
        ClearCanvas();

        if (_chartSeriesList == null || _chartSeriesList.Count == 0 || !_chartSeriesList.Any(s => s.isVisible))
            return;

        float chartWidth = drawer.Width - _paddingLeft - _paddingRight;
        float chartHeight = drawer.Height - _paddingTop - _paddingBottom;

        // Scaling 
        float globalMin = float.MaxValue;
        float globalMax = float.MinValue;
        bool hasVisibleData = false;

        foreach (var series in _chartSeriesList)
        {

            if (!series.isVisible || series.data == null || series.data.Length == 0)
                continue;

            if (series.chartType != ChartSeries.ChartType.GroupBar)
            {
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
            DrawLabels(globalMin, globalMax, chartWidth, chartHeight,
                    _chartSeriesList.Where(s => s.isVisible && s.data != null).Select(s => s.data.Length).DefaultIfEmpty(0).Max());

        for (var index = 0; index < _chartSeriesList.Count; index++)
        {
            var series = _chartSeriesList[index];
            if (!series.isVisible || series.data == null || series.data.Length == 0)
                continue;

            if (series.chartType == ChartSeries.ChartType.Bar)
            {
                var data = series.data;
                var dataCount = series.data.Count();
                for (int i = 0; i < dataCount; i++)
                {
                    float x = _paddingLeft + i / (float)(dataCount - 1) * chartWidth;
                    float y = _paddingBottom + (data[i] - globalMin) / (globalMax - globalMin) * chartHeight;

                    Vector2 start = new Vector2(x, _paddingBottom);
                    Vector2 end = new Vector2(x, y);
                    drawer.DrawLine(start, end, series.axesColor, series.brushSize, _wiggleSize);

                    var label = IsAnyGroupBar() ? series.onXLabelCreated != null ? series.onXLabelCreated(i) : i.ToString() : null;
                    if (label != null)
                    {
                        CreateLabel(label, new(x, y + 20f), series.labelColor, rotation: 0f);
                    }
                }
            }
            else if (series.chartType == ChartSeries.ChartType.Line)
            {
                var data = series.data;
                var dataCount = series.data.Count();
                for (int i = 0; i < dataCount; i++)
                {
                    float x = _paddingLeft + i / (float)(dataCount - 1) * chartWidth;
                    float y = _paddingBottom + (data[i] - globalMin) / (globalMax - globalMin) * chartHeight;

                    if (i == 0 || i == dataCount - 1)
                        continue;

                    float x0 = _paddingLeft + (i - 1) / (float)(dataCount - 1) * chartWidth;
                    float y0 = _paddingBottom + (data[i - 1] - globalMin) / (globalMax - globalMin) * chartHeight;
                    drawer.DrawLine(new Vector2(x0, y0), new Vector2(x, y), series.axesColor, series.brushSize, _wiggleSize);

                    var label = IsAnyGroupBar() ? series.onXLabelCreated != null ? series.onXLabelCreated(i) : i.ToString() : null;
                    if (label != null)
                    {
                        CreateLabel(label, new(x, y + 20f), series.labelColor, rotation: 0f);
                    }
                }
            }
            else if (series.chartType == ChartSeries.ChartType.GroupBar)
            {
                float minVal = series.data.Min();
                float maxVal = series.data.Max();
                float range = Mathf.Max(0.0001f, maxVal - minVal);

                float groupWidth = chartWidth / series.groupCount;
                float barWidth = groupWidth / series.CategoryCount();
                for (int group = 0; group < series.groupCount; group++)
                {
                    for (int cat = 0; cat < series.CategoryCount(); cat++)
                    {
                        int dataIndex = group * series.CategoryCount() + cat;
                        if (dataIndex >= series.data.Length) continue;

                        float value = series.data[dataIndex];

                        float y = 0;
                        if (Mathf.Approximately(maxVal, minVal))
                            y = _paddingBottom + chartHeight * 0.5f;
                        else
                            y = _paddingBottom + (value - minVal) / range * chartHeight;

                        float x = _paddingLeft + group * groupWidth + cat * barWidth;

                        // Color: use per-category if defined, else fallback by group
                        Color barColor = (series.categoryColors != null && series.categoryColors.Length > cat)
                            ? series.categoryColors[group]
                            : (group == 0 ? Color.green : Color.red);

                        // Draw bar
                        drawer.DrawRect(
                            new Vector2(x, _paddingBottom),
                            new Vector2(x + barWidth, y),
                            barColor,
                            1f, _wiggleSize
                        );

                        // Label inside bar
                        if (series.categoryNames != null && cat < series.categoryNames.Length)
                        {
                            Vector2 labelPos = new Vector2(x + barWidth / 2f, y / 2f);
                            CreateLabel(series.categoryNames[cat], labelPos, Color.white, rotation: 90f);
                        }
                    }

                    string groupLabel = (series.groupNames != null && group < series.groupNames.Length)
                        ? series.groupNames[group]
                        : "Group " + (group + 1);

                    Vector2 groupLabelPos = new Vector2(
                        _paddingLeft + group * groupWidth + groupWidth / 2f,
                        _paddingBottom - 20f
                    );
                    CreateLabel(groupLabel, groupLabelPos, series.labelColor, rotation: 0f);
                }
            }
        }
    }

    private bool IsAnyGroupBar()
    {
        return _chartSeriesList.Where((x) => x.chartType == ChartSeries.ChartType.GroupBar && x.isVisible).Count() > 0;
    }

    private void DrawAxes()
    {
        float left = _paddingLeft;
        float right = drawer.Width - _paddingRight;
        float bottom = _paddingBottom;
        float top = drawer.Height - _paddingTop;

        drawer.DrawLine(new Vector2(left, bottom), new Vector2(right, bottom), _gridColour, _brushSize, _wiggleSize);
        drawer.DrawLine(new Vector2(left, bottom), new Vector2(left, top), _gridColour, _brushSize, _wiggleSize);
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

    private List<(int index, float x)> CalculateXAxisGridPoints(int maxLen, float chartWidth, bool withSpacing = false)
    {
        var points = new List<(int index, float x)>();
        if (maxLen <= 0)
            return points;

        for (int i = 0; i < maxLen; i++)
        {
            float x = _paddingLeft + i / (float)(maxLen - 1) * chartWidth;
            points.Add((i, x));
        }
        return points;
    }



    private void DrawGrid(float minVal, float maxVal, float chartWidth, float chartHeight)
    {
        float left = _paddingLeft;
        float right = drawer.Width - _paddingRight;
        float bottom = _paddingBottom;
        float top = drawer.Height - _paddingTop;

        int maxLen = _chartSeriesList.Where(s => s.isVisible && s.data != null).Select(s => s.data.Length).DefaultIfEmpty(0).Max();

        if (maxLen == 0)
            return;

        var yPoints = CalculateYAxisGridPoints(minVal, maxVal, chartHeight);
        foreach (var (value, y) in yPoints)
        {
            drawer.DrawLine(new Vector2(left, y), new Vector2(right, y), _gridColour, 1f, _wiggleSize);
        }

        var xPoints = CalculateXAxisGridPoints(maxLen, chartWidth);
        foreach (var (index, x) in xPoints)
        {
            drawer.DrawLine(new Vector2(x, bottom), new Vector2(x, top), _gridColour, 1f, _wiggleSize);
        }
    }

    private void DrawLabels(float minVal, float maxVal, float chartWidth, float chartHeight, int maxLen)
    {
        // Clean up old labels
        foreach (Transform child in _labelParent)
            Destroy(child.gameObject);

        var visibleSeries = _chartSeriesList.Where(s => s.isVisible && s.data != null && s.data.Length > 0).ToList();

        var yGridPoints = CalculateYAxisGridPoints(minVal, maxVal, chartHeight);
        foreach (var (value, y) in yGridPoints)
        {
            var series = visibleSeries.FirstOrDefault(s => s.data.Contains(value));
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

        // Only labeling x-axis when there's no GroupBar chart
        if (!IsAnyGroupBar())
        {
            var xPoints = CalculateXAxisGridPoints(maxLen, chartWidth);

            foreach (var series in visibleSeries)
            {
                foreach (var (index, x) in xPoints)
                {
                    if (index == 0) continue;

                    string xLabel = series.onXLabelCreated != null
                        ? series.onXLabelCreated(index).ToString()
                        : index.ToString();

                    CreateLabel(xLabel, new Vector2(x, _paddingBottom - _labelOffset), series.labelColor);
                }
            }

            return;
        }

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
        }


        // Always show 0 on X-axis (centered)
        CreateLabel("0", new Vector2(_paddingLeft - _labelOffset, _paddingBottom - _labelOffset));
    }



    private void CreateLabel(string text, Vector2 canvasPos, Color? labelColor = null, float rotation = 0f, int fontSize = 0)
    {
        if (_labelFont == null || _labelParent == null)
            return;

        GameObject labelObj = new GameObject("Label", typeof(RectTransform));
        labelObj.transform.SetParent(_labelParent, false);

        Text label = labelObj.AddComponent<Text>();
        label.text = text;
        label.font = _labelFont;
        label.fontSize = fontSize == 0 ? _fontSize : fontSize;
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

    void OnDestroy()
    {
        drawer.Release();
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

    public System.Func<int, string> onXLabelCreated;

    public System.Func<float, string> onYLabelCreated;
    public System.Func<bool, dynamic> onVisibilityChanged;

    public int groupCount { get; set; }

    public int CategoryCount()
    {
        if (data != null)
        {
            return data.Length / groupCount;
        }
        return 0;
    }

    public string[] groupNames { get; set; }

    public string[] categoryNames { get; set; }

    public Color[] categoryColors { get; set; }

    public static ChartSeries Create(
        string name,
        ChartType type,
        Color? axesColor = null,
        float[] data = null,
        float brushSize = 3f,
        Color? labelColor = null
    )
    {
        var chart = new ChartSeries
        {
            name = name,
            data = data,
            chartType = type,
            axesColor = axesColor ?? Color.white,
            brushSize = brushSize,
            labelColor = labelColor ?? Color.white,

            // Set default formatters
            onXLabelCreated = index => index.ToString(),
            onYLabelCreated = value => value.ToString("0.0")
        };
        return chart;
    }

    public static ChartSeries CreateGroup(
        string name,
        string[] groupNames = null,
        string[] categoryNames = null,
        float brushSize = 3f,
        float[] data = null,
        Color[] categoryColors = null,
        Color? labelColor = null,
        Color? axesColor = null
    )
    {
        var chart = Create(name, ChartType.GroupBar, axesColor, data, brushSize, labelColor);

        chart.groupCount = groupNames.Count();
        chart.groupNames = groupNames;
        chart.categoryNames = categoryNames;

        List<Color> catColors = new();

        if (categoryColors == null)
        {
            for (var i = 0; i < chart.groupCount; i++)
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
