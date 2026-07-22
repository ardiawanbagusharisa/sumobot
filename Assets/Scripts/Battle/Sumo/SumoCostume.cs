
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

        // Appended last on purpose: SumoPart is serialized as an int (e.g. PartSwitcher.part),
        // so new values must go at the end to avoid shifting existing ones (FaceSide stays 3).
        // Body renders via the Robot/Body child that already exists on every bot; it is the
        // bot's chassis, distinct from the FaceSide red/green side marker.
        Body,
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
            {SumoPart.Body, null},
        };

        public Dictionary<SumoPart, Image> ImagePart { private set; get; } = new()
        {
            {SumoPart.Wheel, null},
            {SumoPart.Eye, null},
            {SumoPart.Accessory, null},
            {SumoPart.FaceSide, null},
            {SumoPart.Body, null},
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

            // Color channels are 0..1 floats; the old new Color(0,255,0)/(255,0,0) relied on
            // clamping. Use the named colors so a future alpha/lerp on the side marker is correct.
            if (Side == Placement.Left)
                SpriteRenderers[SumoPart.FaceSide].color = Color.green;
            else if (Side == Placement.Right)
                SpriteRenderers[SumoPart.FaceSide].color = Color.red;
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
            AttachObject(sprites, null);
        }

        public void AttachObject(Dictionary<SumoPart, Sprite> sprites, Dictionary<SumoPart, Color> tints)
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

                // Skin tint (e.g. Accessory color variants). FaceSide is never tinted here —
                // its color is the red/green side marker set by UpdateSideColor.
                if (tints != null && part.Key != SumoPart.FaceSide
                    && tints.TryGetValue(part.Key, out var tint)
                    && SpriteRenderers[part.Key] != null)
                {
                    SpriteRenderers[part.Key].color = tint;
                }
            });
        }
    }


}