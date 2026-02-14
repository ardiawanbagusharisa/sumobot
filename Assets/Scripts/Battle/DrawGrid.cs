using UnityEngine;
using UnityEngine.UI;

[ExecuteAlways]
public class DrawGrid : MonoBehaviour
{
    public enum FitMode { 
        FitHeight, 
        FitWidth, 
        Contain, 
        Cover 
    }

    [Header("Shader & UI")]
    [SerializeField] ComputeShader _drawComputeShader;
    [SerializeField] RawImage _rawImageGrid; 

    [Header("Grid Params (pixels in RT space)")]
    [SerializeField] private float _gridSpacing = 100f;
    [SerializeField] private float _paddingLeft = 0f;
    [SerializeField] private float _paddingRight = 0f;
    [SerializeField] private float _paddingTop = 0f;
    [SerializeField] private float _paddingBottom = 0f;
    [SerializeField] Color _gridColour = Color.white;
    [SerializeField] Color _backgroundColor = new Color(0, 0, 0, 0);
    [Range(1f, 20f)][SerializeField] float _brushSize = 4f;
    [Range(0f, 2f)][SerializeField] float _wiggleSize = 0.0f;

    [Header("Edge Options")]
    [Tooltip("If true, draw explicit border lines on the drawable bounds.")]
    [SerializeField] private bool _includeEdgeLines = true;

    [Header("World Space Fit")]
    [SerializeField] private bool _fitInWorldSpace = true;
    [SerializeField] private FitMode _fitMode = FitMode.Cover; // default to cover full view
    [SerializeField] private float _distanceFromCamera = 1f;  // world units from camera
    [SerializeField] private float _worldMargin = 0f;          // keep 0 for true full screen
    [SerializeField] private string _sortingLayerName = "Default";
    [SerializeField] private int _sortingOrder = -5;

    RenderTexture _canvasGrid;

    void Start()
    {
        if (!Application.isPlaying) return;
        RebuildAndRedraw();
    }

    void RebuildAndRedraw()
    {
        if (_rawImageGrid == null || _drawComputeShader == null) return;

        // Ensure canvas setup
        var canvas = _rawImageGrid.canvas;
        if (canvas == null || canvas.renderMode != RenderMode.WorldSpace)
        {
            Debug.LogWarning("DrawGrid: Canvas should be World Space for full-screen fit.");
        }

        // Choose camera
        Camera cam = (canvas != null && canvas.worldCamera != null) ? canvas.worldCamera
                    : (Camera.main != null ? Camera.main : Camera.current);

        if (canvas != null)
        {
            canvas.worldCamera = cam;
            canvas.overrideSorting = true;
            canvas.sortingLayerName = _sortingLayerName;
            canvas.sortingOrder = _sortingOrder;
        }

        // Force RT size = camera pixelRect (for true screen edge)
        int rtW, rtH;
        if (cam != null)
        {
            rtW = Mathf.Max(1, Mathf.RoundToInt(cam.pixelRect.width));
            rtH = Mathf.Max(1, Mathf.RoundToInt(cam.pixelRect.height));
        }
        else
        {
            // Fallback to RawImage rect if no camera
            var rect = _rawImageGrid.rectTransform.rect;
            rtW = Mathf.Max(1, Mathf.CeilToInt(rect.width));
            rtH = Mathf.Max(1, Mathf.CeilToInt(rect.height));
        }

        InitCanvas(ref _canvasGrid, rtW, rtH);
        _canvasGrid.filterMode = FilterMode.Point;
        _rawImageGrid.texture = _canvasGrid;

        // Fit the RawImage to cover the camera view at the chosen distance
        if (_fitInWorldSpace && canvas != null && cam != null)
        {
            PlaceAndFitWorldSpace(canvas, _rawImageGrid.rectTransform, cam, _distanceFromCamera, _fitMode, _worldMargin);
        }

        // Draw background + grid
        ClearCanvas(_canvasGrid, _backgroundColor);
        DrawArenaGridCenteredAndBorder();
    }

    // ---------- RT Init / Clear ----------

    void InitCanvas(ref RenderTexture canvas, int w, int h)
    {
        if (canvas != null)
        {
            if (canvas.width == w && canvas.height == h && canvas.IsCreated()) return;
            if (canvas.IsCreated()) canvas.Release();
        }

        canvas = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32)
        {
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp
        };
        canvas.Create();
    }

    void ClearCanvas(RenderTexture canvas, Color color)
    {
        if (canvas == null) return;
        int initK = _drawComputeShader.FindKernel("InitBackground");
        _drawComputeShader.SetVector("_BackgroundColour", color);
        _drawComputeShader.SetTexture(initK, "_Canvas", canvas);
        _drawComputeShader.SetInt("_CanvasWidth", canvas.width);
        _drawComputeShader.SetInt("_CanvasHeight", canvas.height);
        _drawComputeShader.Dispatch(initK, Mathf.CeilToInt(canvas.width / 8f), Mathf.CeilToInt(canvas.height / 8f), 1);
    }

    // ---------- Grid Drawing (centered + guaranteed border) ----------

    void DrawArenaGridCenteredAndBorder()
    {
        if (_canvasGrid == null) return;

        // Half-pixel alignment so lines land on pixel centers
        float left = 0.5f + _paddingLeft;
        float right = _canvasGrid.width - 0.5f - _paddingRight;
        float bottom = 0.5f + _paddingBottom;
        float top = _canvasGrid.height - 0.5f - _paddingTop;

        // Clamp in case paddings exceed canvas
        right = Mathf.Max(right, left);
        top = Mathf.Max(top, bottom);

        // Texture center in pixel space (also on pixel centers)
        float cx = Mathf.Clamp(0.5f * _canvasGrid.width, left, right);
        float cy = Mathf.Clamp(0.5f * _canvasGrid.height, bottom, top);

        // Center cross
        DrawLine(new Vector2(left, cy), new Vector2(right, cy), _gridColour);
        DrawLine(new Vector2(cx, bottom), new Vector2(cx, top), _gridColour);

        // Step outwards
        float s = Mathf.Max(1f, _gridSpacing);
        float eps = 0.5f; // tolerance

        // Horizontal lines
        for (int k = 1; ; k++)
        {
            bool drewAny = false;
            float yUp = cy + k * s, yDown = cy - k * s;

            if (yUp <= top)
            {
                bool isEdge = Mathf.Abs(yUp - top) <= eps || Mathf.Abs(yUp - bottom) <= eps;
                if (!_includeEdgeLines && isEdge) { /* skip */ }
                else DrawLine(new Vector2(left, yUp), new Vector2(right, yUp), _gridColour);
                drewAny = true;
            }
            if (yDown >= bottom)
            {
                bool isEdge = Mathf.Abs(yDown - top) <= eps || Mathf.Abs(yDown - bottom) <= eps;
                if (!_includeEdgeLines && isEdge) { /* skip */ }
                else DrawLine(new Vector2(left, yDown), new Vector2(right, yDown), _gridColour);
                drewAny = true;
            }
            if (!drewAny) break;
        }

        // Vertical lines
        for (int k = 1; ; k++)
        {
            bool drewAny = false;
            float xRight = cx + k * s, xLeft = cx - k * s;

            if (xRight <= right)
            {
                bool isEdge = Mathf.Abs(xRight - right) <= eps || Mathf.Abs(xRight - left) <= eps;
                if (!_includeEdgeLines && isEdge) { /* skip */ }
                else DrawLine(new Vector2(xRight, bottom), new Vector2(xRight, top), _gridColour);
                drewAny = true;
            }
            if (xLeft >= left)
            {
                bool isEdge = Mathf.Abs(xLeft - right) <= eps || Mathf.Abs(xLeft - left) <= eps;
                if (!_includeEdgeLines && isEdge) { /* skip */ }
                else DrawLine(new Vector2(xLeft, bottom), new Vector2(xLeft, top), _gridColour);
                drewAny = true;
            }
            if (!drewAny) break;
        }

        // Explicit border lines to guarantee edges show up even if spacing doesn't land on them
        if (_includeEdgeLines)
        {
            DrawLine(new Vector2(left, bottom), new Vector2(right, bottom), _gridColour); // bottom
            DrawLine(new Vector2(left, top), new Vector2(right, top), _gridColour); // top
            DrawLine(new Vector2(left, bottom), new Vector2(left, top), _gridColour); // left
            DrawLine(new Vector2(right, bottom), new Vector2(right, top), _gridColour); // right
        }
    }

    void DrawLine(Vector2 from, Vector2 to, Color color)
    {
        if (_canvasGrid == null) return;
        DispatchDraw(_canvasGrid, from, to, color, _brushSize, _wiggleSize, false, true);
    }

    void DispatchDraw(RenderTexture target, Vector2 from, Vector2 to, Color color, float brushSize, float wiggleSize, bool isEraser, bool mouseDown)
    {
        int kernel = _drawComputeShader.FindKernel("Update");

        _drawComputeShader.SetBool("_MouseDown", mouseDown);
        _drawComputeShader.SetVector("_PreviousMousePosition", new Vector4(from.x, from.y, 0, 0));
        _drawComputeShader.SetVector("_MousePosition", new Vector4(to.x, to.y, 0, 0));
        _drawComputeShader.SetFloat("_BrushSize", brushSize);
        _drawComputeShader.SetFloat("_WiggleSize", wiggleSize);
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

    // ---------- World-space placement & fit ----------

    void PlaceAndFitWorldSpace(Canvas canvas, RectTransform contentRect, Camera cam, float distance, FitMode mode, float marginWorld)
    {
        // position & face
        var t = canvas.transform;
        t.position = cam.transform.position + cam.transform.forward * distance;
        t.rotation = cam.transform.rotation;

        // visible world size at that distance
        Vector2 vis = VisibleWorldSize(cam, distance);
        vis.x = Mathf.Max(0.0001f, vis.x - 2f * marginWorld);
        vis.y = Mathf.Max(0.0001f, vis.y - 2f * marginWorld);

        // content size in its local units (pixels of RawImage rect)
        Vector2 contentSize = contentRect.rect.size;
        if (contentSize.x < 0.001f || contentSize.y < 0.001f)
        {
            // If RectTransform has zero size, sync to RT size
            contentSize = new Vector2(_canvasGrid.width, _canvasGrid.height);
            contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, contentSize.x);
            contentRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, contentSize.y);
        }

        float sx = vis.x / contentSize.x;
        float sy = vis.y / contentSize.y;
        float s = mode switch
        {
            FitMode.FitWidth => sx,
            FitMode.FitHeight => sy,
            FitMode.Contain => Mathf.Min(sx, sy),
            FitMode.Cover => Mathf.Max(sx, sy),
            _ => Mathf.Max(sx, sy),
        };

        // Scale the whole canvas so that content covers the view
        var canvasRect = canvas.GetComponent<RectTransform>();
        canvasRect.localScale = new Vector3(s, s, 1f);
    }

    static Vector2 VisibleWorldSize(Camera cam, float distance)
    {
        if (cam.orthographic)
        {
            float h = cam.orthographicSize * 2f;
            return new Vector2(h * cam.aspect, h);
        }
        float halfH = Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * distance;
        float worldHeight = halfH * 2f;
        return new Vector2(worldHeight * cam.aspect, worldHeight);
    }

    // ---------- Editor hooks ----------

    void OnEnable()
    {
        if (!Application.isPlaying) RebuildAndRedraw();
    }

    void OnValidate()
    {
        RebuildAndRedraw();
    }

    void OnRectTransformDimensionsChange()
    {
        // If this script is on the same object as the RawImage, this will refresh on size drags in the editor.
        if (!Application.isPlaying) RebuildAndRedraw();
    }
}
