using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class ColorPalette
{
    private Texture2D _inputTexture;
    private int _numberOfFixedColors;

    public ColorPalette(Texture2D inputTexture, int numberOfFixedColors = 3)
    {
        _inputTexture = inputTexture;
        _numberOfFixedColors = numberOfFixedColors;
    }

    public List<Color> GenerateColors(int count, bool usePreset = true)
    {
        // If usePreset = true, the first 3 colors will be fixed (black, white, gray)
        if (_inputTexture == null)
        {
            Logger.Error("Input texture is missing for color generation! Returning empty list.");
            return new List<Color>();
        }

        List<Color> generatedColors = new List<Color>();

        if (usePreset) {
            if (_numberOfFixedColors >= 1 && generatedColors.Count < count)
                generatedColors.Add(Color.black);
            if (_numberOfFixedColors >= 2 && generatedColors.Count < count)
                generatedColors.Add(Color.white);
            if (_numberOfFixedColors >= 3 && generatedColors.Count < count)
                generatedColors.Add(Color.gray);
        }

        int start = usePreset ? generatedColors.Count : 0;
        for (int i = start; i < count; i++)
        {
            int randX = Random.Range(0, _inputTexture.width);
            int randY = Random.Range(0, _inputTexture.height);
            generatedColors.Add(_inputTexture.GetPixel(randX, randY));
        }

        if (generatedColors.Count > count)
        {
            return generatedColors.GetRange(0, count);
        }

        return generatedColors;
    }
}
