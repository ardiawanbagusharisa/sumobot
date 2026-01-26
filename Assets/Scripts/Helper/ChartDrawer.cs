using UnityEngine;
using UnityEngine.UI;

public enum DrawBackend
{
    Material, Compute, Dynamic
}

public abstract class Drawer
{
    public static Drawer CreateDrawer(
        RawImage image,
        Material material,
        ComputeShader compute,
        Color background,
        DrawBackend force = DrawBackend.Dynamic)
    {
        bool isSupportCompute = SystemInfo.supportsComputeShaders;
        Logger.Info($"IsComputeShaderSupport: {isSupportCompute}");

        // Force override (e.g., for testing or debugging)
        if (force == DrawBackend.Material)
            return new MaterialDrawer(image, material, background);

        if (force == DrawBackend.Compute)
        {
            // Only use ComputeShader if supported
            if (isSupportCompute)
                return new ComputeShaderDrawer(image, compute, background);
            else
                return new MaterialDrawer(image, material, background);
        }

        // Default: dynamic — use ComputeShader if available, else Material
        if (isSupportCompute)
            return new ComputeShaderDrawer(image, compute, background);
        else
            return new MaterialDrawer(image, material, background);
    }

    public RawImage Image;
    public Color BackgroundColor;
    public int Width;
    public int Height;

    public RenderTexture CanvasRT;
    public RenderTexture TempCanvasRT;
    public ComputeShader Compute;
    public Material Material;
    public abstract void DrawRect(Vector2 min, Vector2 max, Color brushColor, float brushSize, float wiggleSize);
    public abstract void DrawLine(Vector2 from, Vector2 to, Color brushColor, float brushSize, float wiggleSize);
    public abstract void ClearCanvas(Color color);
    public abstract void Init();
    public abstract void Release();
}

public class ComputeShaderDrawer : Drawer
{
    public ComputeShaderDrawer(RawImage image, ComputeShader compute, Color backgroundColor)
    {
        Image = image;
        BackgroundColor = backgroundColor;
        Compute = compute;
    }

    public override void ClearCanvas(Color color)
    {
        int initK = Compute.FindKernel("InitBackground");
        Compute.SetVector("_BackgroundColour", color);
        Compute.SetTexture(initK, "_Canvas", CanvasRT);
        Compute.SetInt("_CanvasWidth", CanvasRT.width);
        Compute.SetInt("_CanvasHeight", CanvasRT.height);
        Compute.Dispatch(initK, Mathf.CeilToInt(CanvasRT.width / 8f), Mathf.CeilToInt(CanvasRT.height / 8f), 1);
    }

    public override void DrawLine(Vector2 from, Vector2 to, Color brushColor, float brushSize, float wiggleSize)
    {
        int updateKernel = Compute.FindKernel("Update");

        Compute.SetBool("_MouseDown", true);
        Compute.SetVector("_PreviousMousePosition", new Vector4(from.x, from.y, 0, 0));
        Compute.SetVector("_MousePosition", new Vector4(to.x, to.y, 0, 0));
        Compute.SetFloat("_BrushSize", brushSize);
        Compute.SetFloat("_WiggleSize", wiggleSize);
        Compute.SetVector("_BrushColour", brushColor);
        Compute.SetInt("_CanvasWidth", CanvasRT.width);
        Compute.SetInt("_CanvasHeight", CanvasRT.height);
        Compute.SetTexture(updateKernel, "_Canvas", CanvasRT);

        Compute.Dispatch(
            updateKernel,
            Mathf.CeilToInt(CanvasRT.width / 8f),
            Mathf.CeilToInt(CanvasRT.height / 8f),
            1
        );
    }

    public override void DrawRect(Vector2 min, Vector2 max, Color brushColor, float brushSize, float wiggleSize)
    {
        int kernel = Compute.FindKernel("Update");

        Compute.SetBool("_UseRect", true);
        Compute.SetVector("_RectMin", new Vector4(min.x, min.y, 0, 0));
        Compute.SetVector("_RectMax", new Vector4(max.x, max.y, 0, 0));
        Compute.SetVector("_BrushColour", brushColor);
        Compute.SetFloat("_BrushSize", brushSize);
        Compute.SetFloat("_WiggleSize", wiggleSize);


        Compute.SetInt("_CanvasWidth", CanvasRT.width);
        Compute.SetInt("_CanvasHeight", CanvasRT.height);
        Compute.SetTexture(kernel, "_Canvas", CanvasRT);

        Compute.Dispatch(
            kernel,
            Mathf.CeilToInt(CanvasRT.width / 8f),
            Mathf.CeilToInt(CanvasRT.height / 8f),
            1
        );

        // Reset flag (so future calls don’t accidentally stay in rect mode)
        Compute.SetBool("_UseRect", false);
    }

    public override void Init()
    {
        Rect rect = Image.rectTransform.rect;
        Width = Mathf.CeilToInt(rect.width);
        Height = Mathf.CeilToInt(rect.height);

        if (CanvasRT == null || CanvasRT.width != Width || CanvasRT.height != Height)
        {
            if (CanvasRT != null)
            {
                CanvasRT.Release();
            }

            CanvasRT = new RenderTexture(Width, Height, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };

            CanvasRT.Create();
            Image.texture = CanvasRT;
        }
    }

    public override void Release()
    {
        if (CanvasRT != null) CanvasRT.Release();
    }
}

public class MaterialDrawer : Drawer
{
    public MaterialDrawer(RawImage image, Material material, Color backgroundColor)
    {
        Image = image;
        Material = material;
        BackgroundColor = backgroundColor;
    }
    public override void ClearCanvas(Color color)
    {
        RenderTexture.active = CanvasRT;
        GL.Clear(true, true, color);
        RenderTexture.active = null;
    }

    public override void DrawLine(Vector2 from, Vector2 to, Color brushColor, float brushSize, float wiggleSize)
    {
        Material.SetFloat("_MouseDown", 1f);
        Material.SetVector("_PreviousMousePosition", new Vector4(from.x, from.y, 0, 0));
        Material.SetVector("_MousePosition", new Vector4(to.x, to.y, 0, 0));
        Material.SetFloat("_BrushSize", brushSize);
        Material.SetFloat("_WiggleSize", wiggleSize);
        Material.SetVector("_BrushColour", brushColor);

        Graphics.Blit(CanvasRT, TempCanvasRT, Material);
        Graphics.Blit(TempCanvasRT, CanvasRT);
    }

    public override void DrawRect(Vector2 min, Vector2 max, Color brushColor, float brushSize, float wiggleSize)
    {
        Material.SetFloat("_UseRect", 1f);
        Material.SetVector("_RectMin", new Vector4(min.x, min.y, 0, 0));
        Material.SetVector("_RectMax", new Vector4(max.x, max.y, 0, 0));
        Material.SetVector("_BrushColour", brushColor);
        Material.SetFloat("_BrushSize", brushSize);
        Material.SetFloat("_WiggleSize", wiggleSize);

        Graphics.Blit(CanvasRT, TempCanvasRT, Material);
        Graphics.Blit(TempCanvasRT, CanvasRT);

        Material.SetFloat("_UseRect", 0f); // reset
    }

    public override void Init()
    {
        Rect rect = Image.rectTransform.rect;
        Width = Mathf.CeilToInt(rect.width);
        Height = Mathf.CeilToInt(rect.height);
        
        if (CanvasRT != null) CanvasRT.Release();
        if (TempCanvasRT != null) TempCanvasRT.Release();

        CanvasRT = new RenderTexture(Width, Height, 0, RenderTextureFormat.ARGB32);
        CanvasRT.Create();

        TempCanvasRT = new RenderTexture(Width, Height, 0, RenderTextureFormat.ARGB32);
        TempCanvasRT.Create();

        ClearCanvas(BackgroundColor);

        Image.texture = CanvasRT;
    }

    public override void Release()
    {
        if (CanvasRT != null) CanvasRT.Release();
        if (TempCanvasRT != null) TempCanvasRT.Release();
    }
}