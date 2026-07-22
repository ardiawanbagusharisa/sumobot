using System.Collections;
using System.Collections.Generic;
using SumoCore;
using SumoServices;
using UnityEngine;

// Live bot preview for the Garage screen (approach A: reuse the real Robot.prefab).
// Drives a SumoCostume instance (UI:0, SpriteRenderer-based — an instance of Robot.prefab)
// with the same EquippedCostumeResolver the battle costume uses, so the preview is exactly
// what a battle would render for the signed-in player. The instance is filmed by a dedicated
// camera into a RenderTexture and shown via a RawImage on the Garage canvas (editor setup).
//
// Unlike PlayerProfile.ApplyEquippedForLocalPlayer there is no opponent/ID matching here —
// Garage is always "the local signed-in player". Equipping/unequipping is handled by
// GarageInventoryController; this class only listens for EquipmentChanged and refreshes,
// playing a short bounce on any part that actually changed.
public class GaragePreviewController : MonoBehaviour
{
    [SerializeField] private SumoCostume previewCostume; // Robot.prefab instance, UI = false

    // Optional. When assigned, the preview camera is driven on-demand: disabled from
    // per-frame rendering and told to Render() only when the panel opens or a part
    // changes (incl. during the bounce). Leave unassigned to keep the camera rendering
    // every frame as before. Filming a tiny RT continuously is cheap, but on-demand also
    // means the camera does nothing while the player is on other menu screens.
    [SerializeField] private Camera previewCamera;

    [Header("Equip bounce")]
    [SerializeField] private float bounceScale = 1.15f;
    [SerializeField] private float bounceDuration = 0.12f;

    private IPlayerDataService subscribed;
    private readonly Dictionary<SumoPart, Sprite> shown = new();
    private readonly Dictionary<Transform, Vector3> baseScale = new();
    private readonly Dictionary<SumoPart, Coroutine> bounces = new();
    private bool initialized;
    private bool renderRequested;

    void OnEnable()
    {
        subscribed = GameServices.PlayerData;
        if (subscribed != null)
            subscribed.EquipmentChanged += Refresh;
        if (previewCamera != null)
            previewCamera.enabled = false; // we drive it via Render() in LateUpdate
        Refresh();
    }

    // Coalesce every render request in a frame into a single manual Render(). Runs only
    // while this component's GameObject is active (i.e. the Garage panel is open).
    void LateUpdate()
    {
        if (renderRequested && previewCamera != null)
        {
            previewCamera.Render();
            renderRequested = false;
        }
    }

    // Ask the on-demand camera to draw once this frame. No-op when previewCamera is
    // unassigned (the camera then renders itself every frame).
    private void RequestRender() => renderRequested = true;

    void OnDisable()
    {
        if (subscribed != null)
        {
            subscribed.EquipmentChanged -= Refresh;
            subscribed = null;
        }
    }

    private void Refresh()
    {
        if (previewCostume == null) return;

        var parts = new Dictionary<SumoPart, Sprite>
        {
            { SumoPart.Wheel, DefaultCostume.SpriteFor(SumoPart.Wheel) },
            { SumoPart.Eye, DefaultCostume.SpriteFor(SumoPart.Eye) },
            { SumoPart.Accessory, DefaultCostume.SpriteFor(SumoPart.Accessory) },
            { SumoPart.Body, DefaultCostume.SpriteFor(SumoPart.Body) },
        };
        var tints = new Dictionary<SumoPart, Color>
        {
            { SumoPart.Wheel, Color.white },
            { SumoPart.Eye, Color.white },
            { SumoPart.Accessory, Color.white },
            { SumoPart.Body, Color.white },
        };

        EquippedCostumeResolver.ApplyEquipped(parts, tints);
        previewCostume.AttachObject(parts, tints);

        // Bounce the parts whose sprite actually changed (skip the first build so opening the
        // panel doesn't pop everything at once).
        if (initialized && isActiveAndEnabled)
        {
            foreach (var kv in parts)
                if (!shown.TryGetValue(kv.Key, out var prev) || prev != kv.Value)
                    PlayBounce(kv.Key);
        }
        foreach (var kv in parts) shown[kv.Key] = kv.Value;
        initialized = true;

        RequestRender(); // panel opened or equipment changed -> draw the new state once
    }

    private void PlayBounce(SumoPart part)
    {
        if (!previewCostume.SpriteRenderers.TryGetValue(part, out var sr) || sr == null) return;

        if (bounces.TryGetValue(part, out var running) && running != null)
            StopCoroutine(running);
        bounces[part] = StartCoroutine(Bounce(sr.transform));
    }

    private IEnumerator Bounce(Transform target)
    {
        // Capture the part's designed scale once (prefab parts aren't all 1,1,1 — e.g. Accessory).
        if (!baseScale.TryGetValue(target, out var basis))
        {
            basis = target.localScale;
            baseScale[target] = basis;
        }

        float t = 0f;
        while (t < bounceDuration)
        {
            t += Time.unscaledDeltaTime;
            float k = Mathf.Clamp01(t / bounceDuration);
            float s = Mathf.Lerp(bounceScale, 1f, k); // ease from bounceScale back to 1
            target.localScale = basis * s;
            RequestRender(); // keep the on-demand camera drawing while the part animates
            yield return null;
        }
        target.localScale = basis;
        RequestRender();
    }
}
