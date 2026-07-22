using UnityEngine;

/// <summary>
/// Uniformly scales a <see cref="content"/> child to fit THIS RectTransform's current size,
/// keeping aspect ratio (letterbox). The content is laid out once at <see cref="designSize"/>
/// and then scaled by min(width/designWidth, height/designHeight) — so everything inside scales
/// together as one block and nothing reflows.
///
/// Use it when a cell/card's available space changes (e.g. a responsive GridLayoutGroup) while
/// its inner layout should stay fixed. Note: if this rect is always the same size as designSize
/// (e.g. a fixed 200x200 grid cell + 200x200 design) the scale is always 1 — it only does
/// something when the available size actually differs from the design size.
/// </summary>
[ExecuteAlways]
[RequireComponent(typeof(RectTransform))]
public class UniformScaleToFit : MonoBehaviour
{
    [Tooltip("The child that gets scaled. Lay all visible content out inside it at designSize, " +
             "with the child anchored + pivoted at center and its size = designSize (not stretched). " +
             "Leave empty to use this object's first child.")]
    [SerializeField] private RectTransform content;

    [Tooltip("The reference size the content was designed at. scale = min(w/designW, h/designH).")]
    [SerializeField] private Vector2 designSize = new(200f, 200f);

    [Tooltip("Only shrink to fit, never enlarge past the design size (cap scale at 1).")]
    [SerializeField] private bool shrinkOnly = false;

    private RectTransform self;

    void OnEnable()
    {
        self = (RectTransform)transform;
        Apply();
    }

    // Fired when this rect is resized (e.g. by the GridLayoutGroup driving the cell).
    void OnRectTransformDimensionsChange()
    {
        if (self == null) self = (RectTransform)transform;
        Apply();
    }

    private void Apply()
    {
        var target = content;
        if (target == null && transform.childCount > 0)
            target = transform.GetChild(0) as RectTransform;
        if (target == null || designSize.x <= 0f || designSize.y <= 0f)
            return;

        Rect r = self.rect;
        float scale = Mathf.Min(r.width / designSize.x, r.height / designSize.y);
        if (shrinkOnly) scale = Mathf.Min(scale, 1f);
        if (scale <= 0f || float.IsNaN(scale) || float.IsInfinity(scale))
            return;

        var s = new Vector3(scale, scale, 1f);
        if (target.localScale != s) // avoid needless writes (keeps the scene from going dirty in edit mode)
            target.localScale = s;
    }
}
