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
    [SerializeField] BrushSlider _brushSizeSlider;
    [SerializeField] BrushSlider _wiggleSldier;
    [Range(1f, 20f)][SerializeField] float _brushSize = 4f;
    [Range(0f, 2f)][SerializeField] float _wiggleSize = 1f;
    [SerializeField] bool _useSoftBrush = true;
    [Range(0f, 1f)][SerializeField] float _softnessLevel = 0.5f;
    [SerializeField] private bool _isEraser = false;

    [Header("UI Elements")]
    [SerializeField] RawImage _rawImage;
    [SerializeField] Button _saveButton;

    RenderTexture _canvasRT;
    Vector4 _previousMousePos;

    void Awake()
    {
        // Hook save button
        _saveButton.onClick.AddListener(SaveCurrentCanvas);
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
        _drawComputeShader.SetBool("_IsEraser", _isEraser);


        _drawComputeShader.Dispatch(updK, Mathf.CeilToInt(cw / 8f),
            Mathf.CeilToInt(ch / 8f),
            1
        );

        _previousMousePos = Input.mousePosition;
    }

    public void SetEraserMode(bool isEraser)
    {
        _isEraser = isEraser;
    }

    public void OnEraserButtonClicked()
    {
        SetEraserMode(true);
    }
    public void OnBrushButtonClicked()
    {
        SetEraserMode(false);
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
}
