using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class DrawManager : MonoBehaviour
{
    [Header("Shader & Colors")]
    [SerializeField] ComputeShader _drawComputeShader;
    [SerializeField] Color _backgroundColour = Color.white;
    [SerializeField] Color _brushColour = Color.black;

    [Header("Brush Settings")]
    [SerializeField] Slider _brushSizeSlider;
    [SerializeField] Slider _wiggleSldier;
    [Range(1f, 20f)][SerializeField] float _brushSize = 4f;
    [Range(0f, 2f)][SerializeField] float _wiggleSize = 1f;
    [SerializeField] private bool _isEraser = false;
    [SerializeField] private bool _isGridOn = true;

    [Header("UI Elements")]
    [SerializeField] RawImage _rawImage;
    [SerializeField] Button _saveButton;
    [SerializeField] Button _eraserButton;
    [SerializeField] Button _gridButton;
    [SerializeField] RectTransform _pointerBrush;

    [Header("Grid Settings")]
    [SerializeField] RawImage _rawImageGrid;
    [SerializeField] private float _gridSpacing = 100f;
    [SerializeField] private float _paddingLeft = 20f;
    [SerializeField] private float _paddingRight = 20f;
    [SerializeField] private float _paddingTop = 20f;
    [SerializeField] private float _paddingBottom = 20f;
    [SerializeField] Color _gridColour = Color.white;

    [Header("Color Palette Integration")]
    [SerializeField] private Texture2D _paletteSourceTexture;
    [SerializeField] private Image _colorPreviewImage;
    [SerializeField] private int _numFixColors = 3;
    [SerializeField] private List<Button> _paletteButtons = new List<Button>();

    ColorPalette _colorPalette;
    Vector4 _previousMousePos;
    bool _isDrawing = false;
    RenderTexture _canvasRT;
    RenderTexture _canvasGrid;

    void Awake()
    {
        _saveButton.onClick.AddListener(SaveCurrentCanvas);
    }

    void Start()
    {
        InitCanvas(ref _canvasRT, _rawImage);
        InitCanvas(ref _canvasGrid, _rawImageGrid);

        _canvasRT.filterMode = FilterMode.Point;
        _canvasGrid.filterMode = FilterMode.Point;

        _rawImage.texture = _canvasRT;
        _rawImageGrid.texture = _canvasGrid;

        ClearCanvas(_canvasRT, _backgroundColour);
        ClearCanvas(_canvasGrid, Color.clear);

        _brushSizeSlider.SetValueWithoutNotify(_brushSize);
        _brushSizeSlider.onValueChanged.AddListener(sz => _brushSize = sz);
        _wiggleSldier.SetValueWithoutNotify(_wiggleSize);
        _wiggleSldier.onValueChanged.AddListener(wz => _wiggleSize = wz);

        _previousMousePos = Input.mousePosition;

        if (_isGridOn) 
            DrawGrid();
        _gridButton.GetComponent<Image>().color = _isGridOn ? Color.white : Color.grey;

        InitPalette();

        _colorPreviewImage.color = _paletteButtons[0].GetComponent<Image>().color;
        SetBrushColor(_colorPreviewImage.color);
    }   

    void Update()
    {
        Vector2 mousePos = Input.mousePosition;
        UpdatePointerPosition(mousePos);

        if (Input.GetMouseButtonUp(0))
        {
            _isDrawing = false;
            return;
        }

        if (Input.GetMouseButtonDown(0))
        {
            if (IsPointerOverUIButNotCanvas())
            {
                _isDrawing = false;
                return;
            }

            _isDrawing = true;
        }

        if (!_isDrawing || !Input.GetMouseButton(0))
        {
            _previousMousePos = mousePos;
            return;
        }

        RectTransformUtility.ScreenPointToLocalPointInRectangle(_rawImage.rectTransform, mousePos, null, out Vector2 localCurr);
        RectTransformUtility.ScreenPointToLocalPointInRectangle(_rawImage.rectTransform, _previousMousePos, null, out Vector2 localPrev);

        var r = _rawImage.rectTransform.rect;
        float cw = _canvasRT.width, ch = _canvasRT.height;

        Vector4 curr = new Vector4((localCurr.x - r.x) / r.width * cw, (localCurr.y - r.y) / r.height * ch, 0, 0);
        Vector4 prev = new Vector4((localPrev.x - r.x) / r.width * cw, (localPrev.y - r.y) / r.height * ch, 0, 0);

        DispatchDraw(_canvasRT, new Vector2(prev.x, prev.y), new Vector2(curr.x, curr.y), _brushColour, _brushSize, _wiggleSize, _isEraser, true);

        _previousMousePos = mousePos;
    }

    private void DispatchDraw(RenderTexture target, Vector2 from, Vector2 to, Color color, float brushSize, float wiggleSize, bool isEraser, bool mouseDown)
    {
        int kernel = _drawComputeShader.FindKernel("Update");

        _drawComputeShader.SetBool("_MouseDown", mouseDown);
        _drawComputeShader.SetVector("_PreviousMousePosition", new Vector4(from.x, from.y, 0, 0));
        _drawComputeShader.SetVector("_MousePosition", new Vector4(to.x, to.y, 0, 0));
        _drawComputeShader.SetFloat("_BrushSize", brushSize);
        _drawComputeShader.SetFloat("_WiggleSize", wiggleSize);

        Color brushColor = _isEraser ? 
            new Color(_backgroundColour.r, _backgroundColour.g, _backgroundColour.b, 0f) : _brushColour;

        _drawComputeShader.SetVector("_BrushColour", brushColor);
        _drawComputeShader.SetVector("_BrushColour", color);
        _drawComputeShader.SetBool("_IsEraser", isEraser);
        _drawComputeShader.SetInt("_CanvasWidth", target.width);
        _drawComputeShader.SetInt("_CanvasHeight", target.height);
        _drawComputeShader.SetTexture(kernel, "_Canvas", target);

        _drawComputeShader.Dispatch(
            kernel,
            Mathf.CeilToInt(target.width / 8f),
            Mathf.CeilToInt(target.height / 8f),
            1
        );
    }

    private bool IsPointerOverUIButNotCanvas()
    {
        PointerEventData eventData = new PointerEventData(EventSystem.current)
        {
            position = Input.mousePosition
        };

        var results = new System.Collections.Generic.List<RaycastResult>();
        EventSystem.current.RaycastAll(eventData, results);

        foreach (var result in results)
        {
            // UI element under pointer, and it's not the canvas
            if (result.gameObject != _rawImage.gameObject)
                return true; 
        }
        return false;
    }

    void InitCanvas(ref RenderTexture canvas, RawImage rawImage)
    {
        var rect = rawImage.rectTransform.rect;
        int w = Mathf.CeilToInt(rect.width);
        int h = Mathf.CeilToInt(rect.height);

        canvas = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32)
        {
            enableRandomWrite = true
        };
        canvas.Create();
    }

    void ClearCanvas(RenderTexture canvas, Color color)
    {
        int initK = _drawComputeShader.FindKernel("InitBackground");
        _drawComputeShader.SetVector("_BackgroundColour", color);
        _drawComputeShader.SetTexture(initK, "_Canvas", canvas);
        _drawComputeShader.SetInt("_CanvasWidth", canvas.width);
        _drawComputeShader.SetInt("_CanvasHeight", canvas.height);
        _drawComputeShader.Dispatch(initK, Mathf.CeilToInt(canvas.width / 8f), Mathf.CeilToInt(canvas.height / 8f), 1);
    }

    public void SwitchEraserMode()
    {
        _isEraser = !_isEraser;
        _eraserButton.GetComponent<Image>().color = _isEraser ? Color.grey : Color.white;
    }

    public void SwitchGrid()
    {
        ClearCanvas(_canvasGrid, Color.clear);

        _isGridOn = !_isGridOn;
        if (_isGridOn)
            DrawGrid();

        _gridButton.GetComponent<Image>().color = _isGridOn ? Color.white : Color.grey;
    }

    public void SetBrushColor(Color brushColor) => _brushColour = brushColor;

    public void UpdatePointerPosition(Vector2 mousePosition)
    {
        if (_pointerBrush != null)
            _pointerBrush.position = mousePosition;
    }

    private void DrawGrid()
    {
        float left = _paddingLeft, right = _canvasGrid.width - _paddingRight;
        float bottom = _paddingBottom, top = _canvasGrid.height - _paddingTop;

        for (int i = 0; i <= Mathf.FloorToInt((top - bottom) / _gridSpacing); i++)
        {
            float y = bottom + i * _gridSpacing;
            if (y >= bottom && y <= top)
                DrawLine(new Vector2(left, y), new Vector2(right, y), _gridColour);
        }

        for (int i = 0; i <= Mathf.FloorToInt((right - left) / _gridSpacing); i++)
        {
            float x = left + i * _gridSpacing;
            if (x >= left && x <= right)
                DrawLine(new Vector2(x, bottom), new Vector2(x, top), _gridColour);
        }
    }

    private void DrawLine(Vector2 from, Vector2 to, Color color)
    {
        DispatchDraw(_canvasGrid, from, to, color, 1f, _wiggleSize, false, true);
    }


    void SaveCurrentCanvas()
    {
        string path = System.IO.Path.Combine(Application.dataPath, "Sprite.png");
        int counter = 1;
        while (System.IO.File.Exists(path))
        {
            path = System.IO.Path.Combine(Application.dataPath, $"Sprite_{counter}.png");
            counter++;
        }

        var prevRT = RenderTexture.active;
        RenderTexture.active = _canvasRT;

        var tex = new Texture2D(_canvasRT.width, _canvasRT.height, TextureFormat.ARGB32, false);
        tex.ReadPixels(new Rect(0, 0, _canvasRT.width, _canvasRT.height), 0, 0);
        tex.Apply();

        Color[] pixels = tex.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i].a = pixels[i].a < 0.05f ? 0f : 1f;
        }
        tex.SetPixels(pixels);
        tex.Apply();

        System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
        Debug.Log($"Saved PNG to: {path}");

        RenderTexture.active = prevRT;
        Destroy(tex);

#if UNITY_EDITOR
        UnityEditor.AssetDatabase.Refresh();
#endif
    }

    // Also called from the button. 
    public void InitPalette()
    {
        if (_paletteSourceTexture == null)
        {
            Debug.LogError("Drawing Color Source Texture is missing! Cannot generate palette.");
            SetBrushColor(Color.black);
            return;
        }

        _colorPalette = new ColorPalette(_paletteSourceTexture, _numFixColors);

        if (_paletteButtons.Count == 0)
        {
            Debug.LogWarning("No color palette buttons assigned. Skipping palette assignment.");
            SetBrushColor(Color.black);
            return;
        }

        List<Color> generatedColors = _colorPalette.GenerateColors(_paletteButtons.Count);

        for (int i = 0; i < _paletteButtons.Count; i++)
        {
            int index = i;
            if (i >= generatedColors.Count) break;

            Image btnImage = _paletteButtons[i].GetComponent<Image>();
            btnImage.color = generatedColors[i];

            _paletteButtons[i].onClick.RemoveAllListeners();
            _paletteButtons[i].onClick.AddListener(() =>
            {
                _colorPreviewImage.color = generatedColors[index];
                SetBrushColor(generatedColors[index]);
                Debug.Log($"Picked color: {_paletteButtons[index]}");
            });
        }

        if (generatedColors.Count > 0) 
            SetBrushColor(generatedColors[0]); 
        else 
            SetBrushColor(Color.black); 

        Debug.Log("Drawing tool palette initialized.");
    }
}
