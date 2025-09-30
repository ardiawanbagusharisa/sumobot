using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class ChartManager : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField] RawImage rawImage;
    [SerializeField] Color backgroundColour = Color.white;
    [SerializeField] Color gridColour = Color.white;
    [Range(1f, 20f)][SerializeField] float brushSize = 3f;
    [Range(0f, 2f)][SerializeField] float wiggleSize = 1f;
    [Range(12f, 32f)][SerializeField] int fontSize = 12;
    [Range(12f, 24f)][SerializeField] int panelFontSize = 14;

    [Header("ChartManager Settings")]
    [SerializeField] private bool enableDebugData = true;
    [SerializeField] private bool enablePallete = true;
    [SerializeField] private bool drawAxes = true;
    [SerializeField] private bool drawGrid = true;
    [SerializeField] private bool drawLabels = true;
    [SerializeField] private Font labelFont;
    [SerializeField] private Transform labelParent;
    [SerializeField] public int XGridSpacing = 10;
    [SerializeField] private float YGridSpacing = 100f;
    [SerializeField] private float paddingLeft = 20f;
    [SerializeField] private float paddingRight = 20f;
    [SerializeField] private float paddingTop = 20f;
    [SerializeField] private float paddingBottom = 20f;
    [SerializeField] private float labelOffset = 10f;

    [Header("SidePanel Settings")]
    [SerializeField] private RectTransform panelParent;
    [SerializeField] private GameObject togglePrefab;
    [SerializeField] private List<ChartSeries> chartSeriesList = new();

    [Header("Chart Colors Source")]
    [SerializeField] private Texture2D palleteSrcTexture;
    [SerializeField] private int numFixColors = 0;
    [SerializeField] private int numSeries = 3;

    [SerializeField] ComputeShader drawShader;
    [SerializeField] Material drawMaterial;
    ColorPalette colorPallete;
    public Drawer drawer;
    public DrawBackend ForceDraw = DrawBackend.Dynamic;

    private void DebugPopulateSeries()
    {
        chartSeriesList.Clear();

        for (int i = 0; i < numSeries; i++)
        {
            float[] data = new float[52];

            // Add 0 values to first and last index. 
            data[0] = data[data.Length - 1] = 0;

            for (int j = 1; j < data.Length - 1; j++)
                data[j] = Random.Range(0f, 50f);

            chartSeriesList.Add(ChartSeries.Create($"Series {i + 1}", ChartSeries.ChartType.Bar, Random.ColorHSV(), data));

        }
    }

    private void Start()
    {
        if (enableDebugData)
            DebugPopulateSeries();
        if (enablePallete)
            InitPalette();
        InitSidePanel();
        DrawChart();
    }

    void OnEnable()
    {
        Init();
    }

    private void Init()
    {
        drawer = Drawer.CreateDrawer(rawImage, drawMaterial, drawShader, backgroundColour, ForceDraw);
        drawer.Init();
    }

    public void InitPalette()
    {
        if (palleteSrcTexture == null || chartSeriesList.Count == 0)
        {
            Logger.Error("Source Texture is missing! Cannot generate palette.");
            return;
        }

        colorPallete = new ColorPalette(palleteSrcTexture, numFixColors);
        List<Color> generatedColors = colorPallete.GenerateColors(chartSeriesList.Count, false);

        if (generatedColors == null || generatedColors.Count == 0)
        {
            Logger.Error("Generated colors list is empty! Cannot apply colors to chart series.");
            return;
        }

        for (int i = 0; i < chartSeriesList.Count; i++)
        {
            chartSeriesList[i].AxesColor = generatedColors[i];
        }

        Logger.Info("Drawing tool palette initialized.");
    }

    public void AddChartSeries(ChartSeries chart, bool sidePanel = false)
    {
        var idx = chartSeriesList.FindIndex((x) => x.Name == chart.Name);
        if (idx == -1)
        {
            if (sidePanel)
                AddSidePanel(chart);
            chartSeriesList.Add(chart);
        }
        else
            UpdateChartSeries(chart);
    }

    public ChartSeries GetChartSeries(string name)
    {
        return chartSeriesList.Find((el) => el.Name == name);
    }
    public void UpdateChartSeries(ChartSeries chart)
    {
        int chartIndex = chartSeriesList.FindIndex((x) => x.Name == chart.Name);
        if (chartIndex != -1)
        {
            bool isCurrentChartVisible = chartSeriesList[chartIndex].IsVisible;
            chartSeriesList[chartIndex] = chart;
            chartSeriesList[chartIndex].IsVisible = isCurrentChartVisible;
        }
    }

    public void InitSidePanel()
    {
        if (panelParent == null || togglePrefab == null || chartSeriesList == null)
            return;

        ClearSidePanels();

        foreach (ChartSeries series in chartSeriesList)
        {
            AddSidePanel(series);
        }

        LayoutRebuilder.ForceRebuildLayoutImmediate(panelParent);
    }
    private void AddSidePanel(ChartSeries series)
    {
        GameObject toggleObj = Instantiate(togglePrefab, panelParent);
        Toggle toggle = toggleObj.GetComponent<Toggle>();
        Text label = toggleObj.GetComponentInChildren<Text>();

        toggle.isOn = series.IsVisible;
        label.text = series.Name;
        label.color = series.LabelColor;
        label.font = labelFont;
        label.fontSize = panelFontSize;

        toggle.onValueChanged.AddListener(isOn =>
        {
            ChartSeries runningChart = chartSeriesList.Find((el) => el.Name == series.Name);
            if (runningChart != null)
            {
                runningChart.IsVisible = isOn;
                DrawChart();
            }

            series.OnVisible.Invoke(isOn);
        });
    }


    public void ClearChartSeries()
    {
        chartSeriesList.Clear();
    }

    public void ClearSidePanels()
    {
        foreach (Transform toggle in panelParent)
            Destroy(toggle.gameObject);
    }

    private void ClearCanvas()
    {
        drawer.ClearCanvas(backgroundColour);
        foreach (Transform child in labelParent)
            Destroy(child.gameObject);
    }

    public void DrawChart()
    {
        ClearCanvas();

        if (chartSeriesList == null || chartSeriesList.Count == 0 || !chartSeriesList.Any(s => s.IsVisible))
            return;

        float chartWidth = drawer.Width - paddingLeft - paddingRight;
        float chartHeight = drawer.Height - paddingTop - paddingBottom;

        // Scaling 
        float globalMin = float.MaxValue;
        float globalMax = float.MinValue;
        bool hasVisibleData = false;

        foreach (var series in chartSeriesList)
        {

            if (!series.IsVisible || series.Data == null || series.Data.Length == 0)
                continue;

            if (series.Type != ChartSeries.ChartType.GroupBar)
            {
                if (series.Data[0] != 0)
                {
                    var temp = series.Data.ToList();
                    temp.Insert(0, 0);
                    series.Data = temp.ToArray();
                }
                if (series.Data.Last() != 0)
                {
                    var temp = series.Data.ToList();
                    temp.Add(0);
                    series.Data = temp.ToArray();
                }
            }

            series.OnPrepareToDraw?.Invoke();

            globalMin = Mathf.Min(globalMin, series.Data.Min());
            globalMax = Mathf.Max(globalMax, series.Data.Max());
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
        if (drawAxes)
            DrawAxes();

        if (drawGrid)
            DrawGrid(globalMin, globalMax, chartWidth, chartHeight);

        if (drawLabels)
            DrawLabels(globalMin, globalMax, chartWidth, chartHeight,
                    chartSeriesList.Where(s => s.IsVisible && s.Data != null).Select(s => s.Data.Length).DefaultIfEmpty(0).Max());

        for (var index = 0; index < chartSeriesList.Count; index++)
        {
            var series = chartSeriesList[index];
            if (!series.IsVisible || series.Data == null || series.Data.Length == 0)
                continue;

            if (series.Type == ChartSeries.ChartType.Bar)
            {
                var data = series.Data;
                var dataCount = series.Data.Count();
                for (int i = 0; i < dataCount; i++)
                {
                    float x = paddingLeft + i / (float)(dataCount - 1) * chartWidth;
                    float y = paddingBottom + (data[i] - globalMin) / (globalMax - globalMin) * chartHeight;

                    Vector2 start = new(x, paddingBottom);
                    Vector2 end = new(x, y);
                    drawer.DrawLine(start, end, series.AxesColor, series.BrushSize, wiggleSize);

                    var label = IsAnyGroupBar() ? series.OnDrawVerticalLabel != null ? series.OnDrawVerticalLabel(i) : i.ToString() : null;
                    if (label != null)
                    {
                        CreateLabel(label, new(x, y + 20f), series.LabelColor, rotation: 0f);
                    }
                }
            }
            else if (series.Type == ChartSeries.ChartType.Line)
            {
                var data = series.Data;
                var dataCount = series.Data.Count();
                for (int i = 0; i < dataCount; i++)
                {
                    float x = paddingLeft + i / (float)(dataCount - 1) * chartWidth;
                    float y = paddingBottom + (data[i] - globalMin) / (globalMax - globalMin) * chartHeight;

                    if (i == 0 || i == dataCount - 1)
                        continue;

                    float x0 = paddingLeft + (i - 1) / (float)(dataCount - 1) * chartWidth;
                    float y0 = paddingBottom + (data[i - 1] - globalMin) / (globalMax - globalMin) * chartHeight;
                    drawer.DrawLine(new Vector2(x0, y0), new Vector2(x, y), series.AxesColor, series.BrushSize, wiggleSize);

                    var label = IsAnyGroupBar() ? series.OnDrawVerticalLabel != null ? series.OnDrawVerticalLabel(i) : i.ToString() : null;
                    if (label != null)
                    {
                        CreateLabel(label, new(x, y + 20f), series.LabelColor, rotation: 0f);
                    }
                }
            }
            else if (series.Type == ChartSeries.ChartType.GroupBar)
            {
                float minVal = series.Data.Min();
                float maxVal = series.Data.Max();
                float range = Mathf.Max(0.0001f, maxVal - minVal);

                float groupWidth = chartWidth / series.GroupCount;
                float barWidth = groupWidth / series.CategoryCount();
                for (int group = 0; group < series.GroupCount; group++)
                {
                    for (int cat = 0; cat < series.CategoryCount(); cat++)
                    {
                        int dataIndex = group * series.CategoryCount() + cat;
                        if (dataIndex >= series.Data.Length) continue;

                        float value = series.Data[dataIndex];

                        float y = 0;
                        if (Mathf.Approximately(maxVal, minVal))
                            y = paddingBottom + chartHeight * 0.5f;
                        else
                            y = paddingBottom + (value - minVal) / range * chartHeight;

                        float x = paddingLeft + group * groupWidth + cat * barWidth;

                        // Color: use per-category if defined, else fallback by group
                        Color barColor = (series.CategoryColors != null && series.CategoryColors.Length > cat)
                            ? series.CategoryColors[group]
                            : (group == 0 ? Color.green : Color.red);

                        // Draw bar
                        drawer.DrawRect(
                            new Vector2(x, paddingBottom),
                            new Vector2(x + barWidth, y),
                            barColor,
                            1f, wiggleSize
                        );

                        // Label inside bar
                        if (series.CategoryNames != null && cat < series.CategoryNames.Length)
                        {
                            Vector2 labelPos = new Vector2(x + barWidth / 2f, y / 2f);
                            CreateLabel(series.CategoryNames[cat], labelPos, Color.white, rotation: 90f);
                        }
                    }

                    string groupLabel = (series.GroupNames != null && group < series.GroupNames.Length)
                        ? series.GroupNames[group]
                        : "Group " + (group + 1);

                    Vector2 groupLabelPos = new Vector2(
                        paddingLeft + group * groupWidth + groupWidth / 2f,
                        paddingBottom - 20f
                    );
                    CreateLabel(groupLabel, groupLabelPos, series.LabelColor, rotation: 0f);
                }
            }
        }
    }

    private bool IsAnyGroupBar()
    {
        return chartSeriesList.Where((x) => x.Type == ChartSeries.ChartType.GroupBar && x.IsVisible).Count() > 0;
    }

    private void DrawAxes()
    {
        float left = paddingLeft;
        float right = drawer.Width - paddingRight;
        float bottom = paddingBottom;
        float top = drawer.Height - paddingTop;

        drawer.DrawLine(new Vector2(left, bottom), new Vector2(right, bottom), gridColour, brushSize, wiggleSize);
        drawer.DrawLine(new Vector2(left, bottom), new Vector2(left, top), gridColour, brushSize, wiggleSize);
    }

    private List<(float value, float y)> CalculateYAxisGridPoints(float minVal, float maxVal, float chartHeight)
    {
        var points = new List<(float value, float y)>();
        int desiredNumYLabels = Mathf.Max(1, Mathf.FloorToInt(chartHeight / YGridSpacing));
        float range = maxVal - minVal;
        float rawStep = range / desiredNumYLabels;

        float p = Mathf.Floor(Mathf.Log10(rawStep));
        float norm = rawStep / Mathf.Pow(10, p);
        float niceN = (norm >= 5) ? 5 : (norm >= 2) ? 2 : 1;
        float actualStep = niceN * Mathf.Pow(10, p);

        float firstVal = Mathf.Ceil(minVal / actualStep) * actualStep;
        for (float val = firstVal; val <= maxVal + actualStep * 0.5f; val += actualStep)
        {
            float y = paddingBottom + (val - minVal) / range * chartHeight;
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
            float x = paddingLeft + i / (float)(maxLen - 1) * chartWidth;
            points.Add((i, x));
        }
        return points;
    }

    private void DrawGrid(float minVal, float maxVal, float chartWidth, float chartHeight)
    {
        float left = paddingLeft;
        float right = drawer.Width - paddingRight;
        float bottom = paddingBottom;
        float top = drawer.Height - paddingTop;

        int maxLen = chartSeriesList.Where(s => s.IsVisible && s.Data != null).Select(s => s.Data.Length).DefaultIfEmpty(0).Max();

        if (maxLen == 0)
            return;

        var yPoints = CalculateYAxisGridPoints(minVal, maxVal, chartHeight);
        foreach (var (value, y) in yPoints)
        {
            drawer.DrawLine(new Vector2(left, y), new Vector2(right, y), gridColour, 1f, wiggleSize);
        }

        var xPoints = CalculateXAxisGridPoints(maxLen, chartWidth);
        foreach (var (index, x) in xPoints)
        {
            drawer.DrawLine(new Vector2(x, bottom), new Vector2(x, top), gridColour, 1f, wiggleSize);
        }
    }

    private void DrawLabels(float minVal, float maxVal, float chartWidth, float chartHeight, int maxLen)
    {
        // Clean up old labels
        foreach (Transform child in labelParent)
            Destroy(child.gameObject);

        var visibleSeries = chartSeriesList.Where(s => s.IsVisible && s.Data != null && s.Data.Length > 0).ToList();

        var yGridPoints = CalculateYAxisGridPoints(minVal, maxVal, chartHeight);
        foreach (var (value, y) in yGridPoints)
        {
            var series = visibleSeries.FirstOrDefault(s => s.Data.Contains(value));
            if (series != null && series.OnDrawHorizontalLabel != null)
            {
                string label = series.OnDrawHorizontalLabel(value);
                CreateLabel(label, new Vector2(paddingLeft - labelOffset, y));
            }
            else
            {
                CreateLabel(value.ToString("0.#"), new Vector2(paddingLeft - labelOffset, y));
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

                    string xLabel = series.OnDrawVerticalLabel != null
                        ? series.OnDrawVerticalLabel(index).ToString()
                        : index.ToString();

                    CreateLabel(xLabel, new Vector2(x, paddingBottom - labelOffset), series.LabelColor);
                }
            }

            return;
        }

        foreach (var series in visibleSeries)
        {
            if (series.Type == ChartSeries.ChartType.GroupBar)
            {
                for (int group = 0; group < series.GroupCount; group++)
                {
                    float groupWidth = chartWidth / series.GroupCount;

                    // Center label under the group
                    float x = paddingLeft + group * groupWidth + groupWidth / 2f;

                    // Use series formatter if provided
                    string groupLabel = series.OnDrawVerticalLabel != null
                        ? series.OnDrawVerticalLabel(group).ToString()
                        : $"G{group + 1}";

                    CreateLabel(groupLabel, new Vector2(x, paddingBottom - labelOffset), series.LabelColor);
                }
            }
        }


        // Always show 0 on X-axis (centered)
        CreateLabel("0", new Vector2(paddingLeft - labelOffset, paddingBottom - labelOffset));
    }

    private void CreateLabel(string text, Vector2 canvasPos, Color? labelColor = null, float rotation = 0f, int fontSize = 0)
    {
        if (labelFont == null || labelParent == null)
            return;

        GameObject labelObj = new GameObject("Label", typeof(RectTransform));
        labelObj.transform.SetParent(labelParent, false);

        Text label = labelObj.AddComponent<Text>();
        label.text = text;
        label.font = labelFont;
        label.fontSize = fontSize == 0 ? this.fontSize : fontSize;
        label.color = labelColor ?? Color.white;
        label.alignment = TextAnchor.MiddleCenter;
        label.horizontalOverflow = HorizontalWrapMode.Overflow;
        label.verticalOverflow = VerticalWrapMode.Overflow;

        RectTransform rt = label.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(100, 30);

        rt.localRotation = Quaternion.Euler(0, 0, rotation);

        RectTransform rawImageRectTransform = rawImage.rectTransform;
        Vector2 rawImageLocalPoint = canvasPos;
        Vector3 worldPointOfCanvasPos = rawImageRectTransform.TransformPoint(rawImageLocalPoint + rawImageRectTransform.rect.min);
        Vector2 screenPoint = RectTransformUtility.WorldToScreenPoint(null, worldPointOfCanvasPos);

        Vector2 localPoint;
        RectTransform labelParentRectTransform = labelParent.GetComponent<RectTransform>();
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

    public string Name;
    public float[] Data; // Flattened array: [val1, val2, ..., valN] where N = groupCount * categoryCount
    public Color AxesColor;
    public Color LabelColor;
    public float BrushSize;
    public bool IsVisible = true;
    public ChartType Type;

    public System.Action OnPrepareToDraw;

    public System.Func<int, string> OnDrawVerticalLabel;

    public System.Func<float, string> OnDrawHorizontalLabel;
    public System.Func<bool, dynamic> OnVisible;

    public int GroupCount { get; set; }

    public int CategoryCount()
    {
        if (Data != null)
        {
            return Data.Length / GroupCount;
        }
        return 0;
    }

    public string[] GroupNames { get; set; }

    public string[] CategoryNames { get; set; }

    public Color[] CategoryColors { get; set; }

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
            Name = name,
            Data = data,
            Type = type,
            AxesColor = axesColor ?? Color.white,
            BrushSize = brushSize,
            LabelColor = labelColor ?? Color.white,

            // Set default formatters
            OnDrawVerticalLabel = index => index.ToString(),
            OnDrawHorizontalLabel = value => value.ToString("0.0")
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

        chart.GroupCount = groupNames.Count();
        chart.GroupNames = groupNames;
        chart.CategoryNames = categoryNames;

        List<Color> catColors = new();

        if (categoryColors == null)
        {
            for (var i = 0; i < chart.GroupCount; i++)
            {
                catColors.Add(Color.white);
            }
            chart.CategoryColors = catColors.ToArray();
        }
        else
        {
            chart.CategoryColors = categoryColors;
        }

        return chart;
    }

}
