using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ColorPaletteGenerator : MonoBehaviour
{
    public Image image;
    public Image colorPreview;  // [Todo] Delete later. 
    public List<Button> colorPalette = new(); 
    private Texture2D inputTexture;
    private DrawManager drawManager;

    public void Start()
    {
        GeneratePalette();
    }

    [ContextMenu("Generate Palette")]
    public void GeneratePalette()
    {
        inputTexture = image.mainTexture as Texture2D;

        if (image == null)
        {
            Debug.LogError("Input texture is missing!");
            return;
        }

        if (colorPalette.Count != 28)
        {
            Debug.LogError("You must assign exactly 28 buttons.");
            return;
        }

        // Set the first 3 fixed colors
        Color[] paletteColors = new Color[28];
        paletteColors[0] = Color.black;
        paletteColors[1] = Color.white;
        paletteColors[2] = Color.gray;

        for (int i = 3; i < 28; i++)
        {
            int randX = Random.Range(0, inputTexture.width);
            int randY = Random.Range(0, inputTexture.height);
            paletteColors[i] = inputTexture.GetPixel(randX, randY);
        }

        // Apply colors to buttons and register click events
        for (int i = 0; i < 28; i++)
        {
            int index = i; // capture for lambda
            Image btnImage = colorPalette[i].GetComponent<Image>();
            btnImage.color = paletteColors[i];

            colorPalette[i].onClick.RemoveAllListeners(); 
            colorPalette[i].onClick.AddListener(() =>
            {
                if (colorPreview != null)
                {
                    colorPreview.color = paletteColors[index];
                }

                drawManager = FindFirstObjectByType<DrawManager>();
                if (drawManager != null)
                {
                    drawManager.SetBrushColor(paletteColors[index]);
                }
                Debug.Log($"Picked color: {paletteColors[index]}");
            });
        }

        Debug.Log("New palette generated.");
    }

    // [Note] Unused for now. 
    public void ColorPick(Button button)
    {
        if (colorPreview != null)
        {
            Color pickedColor = colorPreview.color;
            colorPreview.color = button.GetComponent<Image>().color;
            Debug.Log($"Picked color: {pickedColor}");
        }
    }
}
