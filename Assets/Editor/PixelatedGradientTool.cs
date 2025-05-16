using UnityEngine;
using UnityEditor;
using System.IO;

public class PixelatedGradientTool : EditorWindow
{
    private Color startColor = Color.blue;
    private Color endColor = Color.cyan;
    private int textureWidth = 256;
    private int textureHeight = 256;
    private int pixelSize = 8;
    private int cornerRadius = 0;
    private Texture2D generatedTexture;
    private bool generateButton = false;
    private Color buttonColor = Color.white;
    private GradientType gradientType = GradientType.Linear;
    private GradientAlignment startAlignment = GradientAlignment.TopLeft;
    private GradientAlignment endAlignment = GradientAlignment.BottomRight;

    private enum GradientType
    {
        Linear,
        Radial
    }

    private enum GradientAlignment
    {
        TopLeft,
        TopCenter,
        TopRight,
        CenterLeft,
        Center,
        CenterRight,
        BottomLeft,
        Bottom,
        BottomRight
    }

    [MenuItem("Tools/Pixelated Texture Generator")]
    public static void ShowWindow()
    {
        GetWindow<PixelatedGradientTool>("Pixelated Texture Generator");
    }

    private void OnGUI()
    {
        GUILayout.Label("Gradient Settings", EditorStyles.boldLabel);

        gradientType = (GradientType)EditorGUILayout.EnumPopup("Gradient Type", gradientType);
        startColor = EditorGUILayout.ColorField("Start Color", startColor);
        startAlignment = (GradientAlignment)EditorGUILayout.EnumPopup("Start Alignment", startAlignment);
        endColor = EditorGUILayout.ColorField("End Color", endColor);
        endAlignment = (GradientAlignment)EditorGUILayout.EnumPopup("End Alignment", endAlignment);

        textureWidth = EditorGUILayout.IntField("Texture Width", textureWidth);
        textureHeight = EditorGUILayout.IntField("Texture Height", textureHeight);
        pixelSize = EditorGUILayout.IntField("Pixel Size", pixelSize);
        cornerRadius = EditorGUILayout.IntField("Corner Radius", cornerRadius);

        if (GUILayout.Button("Generate Gradient Texture"))
        {
            generateButton = false;
            GenerateGradientTexture();
        }

        GUILayout.Space(10);
        GUILayout.Label("Basic Pixelated Button Settings", EditorStyles.boldLabel);
        buttonColor = EditorGUILayout.ColorField("Button Color", buttonColor);

        if (GUILayout.Button("Generate Button Texture"))
        {
            generateButton = true;
            GenerateButtonTexture();
        }

        if (generatedTexture != null)
        {
            GUILayout.Label("Generated Texture Preview:");
            GUILayout.Label(generatedTexture, GUILayout.Width(256), GUILayout.Height(256));

            if (GUILayout.Button("Save as PNG"))
            {
                SaveTextureAsPNG();
            }
        }
    }

    private void GenerateGradientTexture()
    {
        generatedTexture = new Texture2D(textureWidth, textureHeight);

        if (gradientType == GradientType.Linear)
        {
            Vector2 startPoint = GetAlignmentPoint(startAlignment);
            Vector2 endPoint = GetAlignmentPoint(endAlignment);

            for (int y = 0; y < textureHeight; y += pixelSize)
            {
                for (int x = 0; x < textureWidth; x += pixelSize)
                {
                    float t = Mathf.InverseLerp(startPoint.y, endPoint.y, y) * 0.5f + Mathf.InverseLerp(startPoint.x, endPoint.x, x) * 0.5f;
                    Color pixelColor = Color.Lerp(startColor, endColor, t);

                    SetPixelBlock(x, y, pixelColor);
                }
            }
        }
        else if (gradientType == GradientType.Radial)
        {
            Vector2 center = new Vector2(textureWidth / 2, textureHeight / 2);
            float maxDistance = Vector2.Distance(Vector2.zero, center);

            for (int y = 0; y < textureHeight; y += pixelSize)
            {
                for (int x = 0; x < textureWidth; x += pixelSize)
                {
                    float distance = Vector2.Distance(new Vector2(x, y), center) / maxDistance;
                    Color pixelColor = Color.Lerp(startColor, endColor, distance);

                    SetPixelBlock(x, y, pixelColor);
                }
            }
        }

        generatedTexture.Apply();
    }

    private void SetPixelBlock(int x, int y, Color color)
    {
        for (int py = 0; py < pixelSize && y + py < textureHeight; py++)
        {
            for (int px = 0; px < pixelSize && x + px < textureWidth; px++)
            {
                if (!IsWithinCornerRadius(x + px, y + py))
                    generatedTexture.SetPixel(x + px, y + py, color);
            }
        }
    }

    private bool IsWithinCornerRadius(int x, int y)
    {
        if (cornerRadius <= 0) return false;
        if ((x < cornerRadius && y < cornerRadius && Vector2.Distance(new Vector2(x, y), new Vector2(cornerRadius, cornerRadius)) > cornerRadius) ||
            (x > textureWidth - cornerRadius && y < cornerRadius && Vector2.Distance(new Vector2(x, y), new Vector2(textureWidth - cornerRadius, cornerRadius)) > cornerRadius) ||
            (x < cornerRadius && y > textureHeight - cornerRadius && Vector2.Distance(new Vector2(x, y), new Vector2(cornerRadius, textureHeight - cornerRadius)) > cornerRadius) ||
            (x > textureWidth - cornerRadius && y > textureHeight - cornerRadius && Vector2.Distance(new Vector2(x, y), new Vector2(textureWidth - cornerRadius, textureHeight - cornerRadius)) > cornerRadius))
        {
            return true;
        }
        return false;
    }

    private Vector2 GetAlignmentPoint(GradientAlignment alignment)
    {
        switch (alignment)
        {
            case GradientAlignment.TopLeft: return new Vector2(0, textureHeight);
            case GradientAlignment.TopCenter: return new Vector2(textureWidth / 2, textureHeight);
            case GradientAlignment.TopRight: return new Vector2(textureWidth, textureHeight);
            case GradientAlignment.CenterLeft: return new Vector2(0, textureHeight / 2);
            case GradientAlignment.Center: return new Vector2(textureWidth / 2, textureHeight / 2);
            case GradientAlignment.CenterRight: return new Vector2(textureWidth, textureHeight / 2);
            case GradientAlignment.BottomLeft: return new Vector2(0, 0);
            case GradientAlignment.Bottom: return new Vector2(textureWidth / 2, 0);
            case GradientAlignment.BottomRight: return new Vector2(textureWidth, 0);
            default: return new Vector2(0, textureHeight);
        }
    }

    private void GenerateButtonTexture()
    {
        generatedTexture = new Texture2D(textureWidth, textureHeight);

        for (int y = 0; y < textureHeight; y += pixelSize)
        {
            for (int x = 0; x < textureWidth; x += pixelSize)
            {
                for (int py = 0; py < pixelSize && y + py < textureHeight; py++)
                {
                    for (int px = 0; px < pixelSize && x + px < textureWidth; px++)
                    {
                        generatedTexture.SetPixel(x + px, y + py, buttonColor);
                    }
                }
            }
        }

        generatedTexture.Apply();
    }

    private void SaveTextureAsPNG()
    {
        byte[] bytes = generatedTexture.EncodeToPNG();
        string defaultFileName = generateButton ? "PixelatedButton.png" : "PixelatedGradient.png";
        string path = EditorUtility.SaveFilePanel("Save Texture As PNG", "", defaultFileName, "png");

        if (!string.IsNullOrEmpty(path))
        {
            File.WriteAllBytes(path, bytes);
            AssetDatabase.Refresh();
            Debug.Log("Texture saved to: " + path);
        }
    }
}