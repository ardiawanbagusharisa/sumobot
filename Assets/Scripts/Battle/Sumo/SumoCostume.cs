
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
    [Serializable]
    public class SumoCostume : MonoBehaviour
    {
        #region Costume properties
        public const string Triangle = "Triangle";
        public const string Wheel = "Wheel";
        public const string Body = "Body";
        public const string Eye = "Eye";
        public const string Accessory = "Accessory";
        public Dictionary<string, SpriteRenderer> SpriteParts { private set; get; } = new()
        {
            {Triangle, null},
            {Wheel, null},
            {Body, null},
            {Eye, null},
            {Accessory, null},
        };

        public Dictionary<string, Image> ImagePart { private set; get; } = new()
        {
            {Triangle, null},
            {Wheel, null},
            {Body, null},
            {Eye, null},
            {Accessory, null},
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

                SpriteParts.ToList().ForEach((part) =>
                {
                    SpriteRenderer sprite = sprites.FirstOrDefault((x) => x.gameObject.CompareTag($"Robot/{part.Key}"));
                    if (sprite != null)
                    {
                        SpriteParts[part.Key] = sprite;
                    }
                });
            }
        }

        public void UpdateSideColor()
        {
            if (UI)
                return;

            if (Side == Placement.Left)
                SpriteParts[Triangle].color = new Color(0, 255, 0);
            else if (Side == Placement.Right)
                SpriteParts[Triangle].color = new Color(255, 0, 0);
        }

        public void AttachToHUD(SumoCostume robotCostume)
        {
            if (!UI)
                return;

            robotCostume.SpriteParts.ToList().ForEach((x) =>
            {
                if (ImagePart.ContainsKey(x.Key) && ImagePart[x.Key] != null)
                {
                    ImagePart[x.Key].sprite = x.Value.sprite;
                    ImagePart[x.Key].color = x.Value.color;
                    ImagePart[x.Key].material = x.Value.material;
                }
            });
        }
    }


}