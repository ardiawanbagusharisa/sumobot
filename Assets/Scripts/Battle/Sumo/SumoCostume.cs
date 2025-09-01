
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace SumoCore
{
    public enum Placement
    {
        Left,
        Right,
        Winner,
    }

    public enum SumoPart
    {
        // Mandatory
        Wheel,
        Eye,
        Accessory,
        // Mandatory

        FaceSide,
    }

    [Serializable]
    public class SumoCostume : MonoBehaviour
    {
        #region Costume properties
        public Dictionary<SumoPart, SpriteRenderer> SpriteRenderers { private set; get; } = new()
        {
            {SumoPart.Wheel, null},
            {SumoPart.Eye, null},
            {SumoPart.Accessory, null},
            {SumoPart.FaceSide, null},
        };

        public Dictionary<SumoPart, Image> ImagePart { private set; get; } = new()
        {
            {SumoPart.Wheel, null},
            {SumoPart.Eye, null},
            {SumoPart.Accessory, null},
            {SumoPart.FaceSide, null},
        };

        public Placement Side;
        public bool UI = true;
        #endregion

        void OnEnable()
        {
            // Used for HUD
            if (UI)
            {
                List<Image> images = GetComponentsInChildren<Image>().ToList();

                ImagePart.ToList().ForEach((part) =>
                {
                    Image image = images.FirstOrDefault((x) => x.gameObject.CompareTag($"Robot/{part.Key}"));
                    if (image != null)
                    {
                        ImagePart[part.Key] = image;
                    }
                });
            }
            else
            // Used for Robot Environtment
            {
                List<SpriteRenderer> sprites = GetComponentsInChildren<SpriteRenderer>().ToList();

                SpriteRenderers.ToList().ForEach((part) =>
                {
                    SpriteRenderer sprite = sprites.FirstOrDefault((x) => x.gameObject.CompareTag($"Robot/{part.Key}"));
                    if (sprite != null)
                    {
                        SpriteRenderers[part.Key] = sprite;
                    }
                });
            }
        }

        public void UpdateSideColor()
        {
            if (UI)
                return;

            if (Side == Placement.Left)
                SpriteRenderers[SumoPart.FaceSide].color = new Color(0, 255, 0);
            else if (Side == Placement.Right)
                SpriteRenderers[SumoPart.FaceSide].color = new Color(255, 0, 0);
        }

        public void AttachToUI(SumoCostume robotCostume)
        {
            if (!UI)
                return;

            robotCostume.SpriteRenderers.ToList().ForEach((x) =>
            {
                if (ImagePart.ContainsKey(x.Key) && ImagePart[x.Key] != null)
                {
                    ImagePart[x.Key].sprite = x.Value.sprite;
                    ImagePart[x.Key].color = x.Value.color;
                    ImagePart[x.Key].material = x.Value.material;
                }
            });
        }

        public void AttachObject(Dictionary<SumoPart, Sprite> sprites)
        {
            if (UI)
                return;

            SpriteRenderers.ToList().ForEach((part) =>
            {
                sprites.TryGetValue(part.Key, out var sprite);
                if (sprite != null)
                {
                    SpriteRenderers[part.Key].sprite = sprite;
                }
            });
        }
    }


}