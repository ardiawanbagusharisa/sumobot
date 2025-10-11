using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class VFXManager : MonoBehaviour
{
    public static VFXManager Instance { get; private set; }

    private bool isActive => Instance != null && gameObject != null && gameObject.activeSelf;

    [System.Serializable]
    public struct FloatRange
    {
        public float Min, Max;
        public FloatRange(float min, float max) { Min = min; Max = max; }
        public float Sample(System.Random rng) => (float)(Min + (Max - Min) * rng.NextDouble());
    }

    [System.Serializable]
    public struct SizeRange
    {
        public float Min, Max;
        public SizeRange(float min, float max) { Min = min; Max = max; }
        public float Sample(System.Random rng) => (float)(Min + (Max - Min) * rng.NextDouble());
    }

    [System.Serializable]
    public class PixelBurstSettings
    {
        [Tooltip("Unique key for cooldown tracking per Sumo")]
        public string Key = "unset";

        [Tooltip("How many particles to spawn per burst/tick")]
        public int Count = 8;

        [Tooltip("Particle size range (world units)")]
        public SizeRange Size = new SizeRange(0.05f, 0.08f);

        [Tooltip("Particle lifetime (seconds)")]
        public FloatRange Lifetime = new FloatRange(0.25f, 0.40f);

        [Tooltip("Initial speed (world units / sec)")]
        public FloatRange Speed = new FloatRange(0.6f, 1.2f);

        [Tooltip("Spread around the bias direction (degrees)")]
        public float SpreadDeg = 70f;

        [Tooltip("Gravity along Y (2D up is +Y)")]
        public float GravityY = 0f;

        [Tooltip("Color over particle [0..1]")]
        public Gradient Color;

        [Tooltip("Cooldown between spawns per Sumo (seconds)")]
        public float Cooldown = 0.05f;
    }

    public enum VFXParentMode
    {
        SumoController = 0,
        VFXManager = 1,
        WorldSpace = 2,
    }

    // ------------------- General -------------------
    [Header("Parenting / Rendering")]
    public VFXParentMode parentMode = VFXParentMode.SumoController;

    [Tooltip("Name of the child created on the Sumo when using UnderSumoVFXRoot.")]
    public string vfxRootName = "VFXRoot";

    [Tooltip("Unity layer for spawned VFX.")]
    public int vfxLayer = 0;

    [Tooltip("SpriteRenderer/LineRenderer sorting order for VFX.")]
    public int vfxSortingOrder = 50;

    [Tooltip("Sorting layer name for LineRenderers.")]
    public string vfxSortingLayerName = "Default";

    [Header("Anchors")]
    [Tooltip("Child path under each Sumo for the left trail anchor.")]
    public string leftTrailPath = "VFX/LeftTrail";
    [Tooltip("Child path under each Sumo for the right trail anchor.")]
    public string rightTrailPath = "VFX/RightTrail";

    // small back/side nudges (in world space) when emitting at anchors
    [Tooltip("Backward offset from anchor for accel/turn puffs.")]
    public float backOffset = 0f;
    [Tooltip("Side displacement for turn puffs (uses sign).")]
    public float turnSideOffset = 0f;


    //// ------------------- Wheel emission -------------------
    //[Header("Wheel Emission (local offsets)")]
    //[Tooltip("If true, accel/turn trails spawn at two wheel-local offsets.")]
    //public bool emitAtTwoWheels = true;
    //[Tooltip("Local-space offset (left wheel) relative to Sumo root (x right, y up).")]
    //public Vector2 leftWheelLocalOffset = new Vector2(0f, 0f);
    //[Tooltip("Local-space offset (right wheel) relative to Sumo root (x right, y up).")]
    //public Vector2 rightWheelLocalOffset = new Vector2(0f, 0f);

    [Header("Acceleration Trail")]
    public PixelBurstSettings Accel = new PixelBurstSettings
    {
        Key = "accel",
        Count = 6,
        Lifetime = new FloatRange(0.25f, 0.40f),
        Speed = new FloatRange(0.6f, 1.2f),
        SpreadDeg = 70f,
        GravityY = 0f,
        Size = new SizeRange(0.05f, 0.08f),
        Cooldown = 0.1f
    };

    [Header("Turn Trail")]
    public PixelBurstSettings Turn = new PixelBurstSettings
    {
        Key = "turn",
        Count = 7,
        Lifetime = new FloatRange(0.25f, 0.42f),
        Speed = new FloatRange(0.5f, 1.0f),
        SpreadDeg = 90f,
        GravityY = 0f,
        Size = new SizeRange(0.05f, 0.08f),
        Cooldown = 0.1f
    };

    [Header("Dash Burst")]
    public PixelBurstSettings Dash = new PixelBurstSettings
    {
        Key = "dash",
        Count = 22,
        Lifetime = new FloatRange(0.45f, 0.65f),
        Speed = new FloatRange(1.4f, 3.2f),
        SpreadDeg = 140f,
        GravityY = 0f,
        Size = new SizeRange(0.06f, 0.10f),
        Cooldown = 0.60f
    };

    [Header("Collision Sparks")]
    public PixelBurstSettings SlowCollision = new PixelBurstSettings
    {
        Key = "col_slow",
        Count = 12,
        Lifetime = new FloatRange(0.30f, 0.40f),
        Speed = new FloatRange(2.0f, 5.0f),
        SpreadDeg = 180f,
        GravityY = -9.0f,
        Size = new SizeRange(0.07f, 0.07f),
        Cooldown = 0.1f
    };

    public PixelBurstSettings FastCollision = new PixelBurstSettings
    {
        Key = "col_fast",
        Count = 30,
        Lifetime = new FloatRange(0.30f, 0.40f),
        Speed = new FloatRange(4.0f, 7.0f),
        SpreadDeg = 180f,
        GravityY = -9.0f,
        Size = new SizeRange(0.07f, 0.07f),
        Cooldown = 0.1f
    };

    [Tooltip("Relative speed threshold to pick the 'fast' burst")]
    public float fastCollisionSpeedThreshold = 3.0f;

    [Tooltip("Spark color is blended by relative speed (0..1)")]
    public Gradient sparkColorOverSpeed;

    // ------------------- Internals -------------------
    private readonly Dictionary<string, float> _nextAllowed = new Dictionary<string, float>();
    private System.Random _rng;
    private PixelParticlePool _pixelPool;
    private Material _lineMatDefault;

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;

        _rng = new System.Random();

        // ensure gradients
        if (Accel.Color == null)
        {
            Accel.Color = new Gradient
            {
                colorKeys = new[]
                {
                    new GradientColorKey(Color.white, 0f),
                    new GradientColorKey(Color.white, 1f),
                },
                alphaKeys = new[]
                {
                    new GradientAlphaKey(0.95f, 0f),
                    new GradientAlphaKey(0.95f, 1f),
                }
            };
        }
        if (Turn.Color == null)
        {
            Turn.Color = new Gradient
            {
                colorKeys = new[]
                {
                    new GradientColorKey(new Color(0.72f,0.72f,0.72f), 0f),
                    new GradientColorKey(new Color(0.75f,0.75f,0.75f), 1f),
                },
                alphaKeys = new[]
                {
                    new GradientAlphaKey(0.9f, 0f),
                    new GradientAlphaKey(0.9f, 1f),
                }
            };
        }
        if (Dash.Color == null)
        {
            Dash.Color = new Gradient
            {
                colorKeys = new[]
                {
                    new GradientColorKey(new Color(0.9f,0.9f,0.9f), 0f),
                    new GradientColorKey(new Color(0.8f,0.8f,0.8f), 1f),
                },
                alphaKeys = new[]
                {
                    new GradientAlphaKey(0.85f, 0f),
                    new GradientAlphaKey(0.85f, 1f),
                }
            };
        }
        if (sparkColorOverSpeed == null)
        {
            sparkColorOverSpeed = new Gradient
            {
                colorKeys = new[]
                {
                    new GradientColorKey(new Color(1f, 0.85f, 0.1f), 0f),
                    new GradientColorKey(new Color(1f, 0.35f, 0.1f), 1f)
                },
                alphaKeys = new[]
                {
                    new GradientAlphaKey(1f, 0f),
                    new GradientAlphaKey(1f, 1f)
                }
            };
        }

        _pixelPool = new PixelParticlePool(capacity: 256, vfxLayer: vfxLayer, sortingOrder: vfxSortingOrder);
    }

    private readonly Dictionary<int, (Transform left, Transform right)> _anchorCache
    = new Dictionary<int, (Transform left, Transform right)>();

    private (Transform left, Transform right) GetTrailAnchors(Transform sumo)
    {
        int id = sumo.GetInstanceID();
        if (_anchorCache.TryGetValue(id, out var pair)) return pair;

        Transform left = sumo.Find(leftTrailPath);
        Transform right = sumo.Find(rightTrailPath);
        _anchorCache[id] = (left, right);
        return (left, right);
    }

    private Vector2 GetWheelWorldPos(Transform sumo, bool isLeft, Vector2 fallbackLocalOffset)
    {
        var (left, right) = GetTrailAnchors(sumo);
        Transform t = isLeft ? left : right;
        if (t != null) return t.position;
        // fallback to your legacy local offsets if anchor missing
        return (Vector2)sumo.TransformPoint(fallbackLocalOffset);
    }

    // ------------------- Public API -------------------

    public void PlayAccelerationTrail(Transform sumo, Vector2 facingDir)
    {
        if (!isActive) return;
        if (!CheckCooldown(Key(sumo, Accel.Key), Accel.Cooldown))
            return;

        EmitAccelAtOffset(sumo, facingDir, Vector2.zero, 0);
        EmitAccelAtOffset(sumo, facingDir, Vector2.zero, 1);
    }

    private void EmitAccelAtOffset(Transform sumo, Vector2 facingDir, Vector2 localOffset, int wheelIndex)
    {
        Transform parent = GetVFXParent(sumo);
        bool isLeft = wheelIndex == 0;
        Vector2 basePos = GetWheelWorldPos(sumo, isLeft, Vector2.zero);
        Vector2 origin = basePos - facingDir.normalized * backOffset;

        EmitPixelBurst(parent, origin, -facingDir, Accel);
    }

    public void PlayTurnTrail(Transform sumo, Vector2 facingDir, int dirSign = 1)
    {
        if (!isActive) return;
        if (!CheckCooldown(Key(sumo, Turn.Key), Turn.Cooldown))
            return;

        EmitTurnAtOffset(sumo, facingDir, Vector2.zero, true, 0);
        EmitTurnAtOffset(sumo, facingDir, Vector2.zero, true, 1);
    }

    private void EmitTurnAtOffset(Transform sumo, Vector2 facingDir, Vector2 localOffset, bool darker, int wheelIndex)
    {
        Transform parent = GetVFXParent(sumo);
        bool isLeft = wheelIndex == 0;

        Vector2 anchorPos = GetWheelWorldPos(sumo, isLeft, Vector2.zero);
        Vector2 worldSide = new Vector2(facingDir.y, -facingDir.x).normalized * turnSideOffset * (isLeft ? -1f : 1f); // [Todo] * Mathf.Sign(dirSign);
        Vector2 worldBack = -facingDir.normalized * backOffset;
        Vector2 worldPos = anchorPos + worldSide + worldBack;

        EmitPixelBurst(parent, worldPos, -facingDir, Turn);
    }

    public void PlayDash(Transform sumo, Vector2 facingDir)
    {
        if (!isActive) return;
        if (!CheckCooldown(Key(sumo, Dash.Key), Dash.Cooldown))
            return;

        Transform parent = GetVFXParent(sumo);
        Transform vfxRoot = sumo.Find("VFX");

        Vector2 basePos = vfxRoot != null ? (Vector2)vfxRoot.position : (Vector2)sumo.position;

        // Guard: if passed-in facing is zero, use the robot's current up
        if (facingDir.sqrMagnitude < 1e-6f)
            facingDir = sumo.up;  // same as (Quaternion.Euler(0,0,rotation)*Vector2.up)

        Vector2 origin = basePos - facingDir.normalized * backOffset;

        EmitPixelBurst(parent, origin, -facingDir, Dash);
    }

    public void PlayCollisionSpark(Vector2 worldPos, float relativeSpeed)
    {
        if (!isActive) return;
        bool fast = relativeSpeed >= fastCollisionSpeedThreshold;
        var s = fast ? FastCollision : SlowCollision;

        float t = Mathf.InverseLerp(fastCollisionSpeedThreshold * 0.5f, fastCollisionSpeedThreshold * 2.0f, relativeSpeed);
        Color blended = sparkColorOverSpeed.Evaluate(Mathf.Clamp01(t));

        // no parent => these are world-space bursts
        // dirBias -> radial; spread=180 makes it omni
        EmitPixelBurst(parent: GetVFXParent(null), origin: worldPos, dirBias: Vector2.up, s: s, overrideColor: blended);
    }

    private bool CheckCooldown(string key, float cd)
    {
        float now = Time.time;
        if (_nextAllowed.TryGetValue(key, out float t) && t > now)
            return false;
        _nextAllowed[key] = now + cd;
        return true;
    }

    private string Key(Transform t, string vfx) => $"{t.GetInstanceID()}::{vfx}";

    private Transform GetVFXParent(Transform sumo)
    {
        switch (parentMode)
        {
            case VFXParentMode.VFXManager:
                return this.transform;

            case VFXParentMode.WorldSpace:
                // no parent => return null so EmitPixelBurst keeps world-space
                return null;

            case VFXParentMode.SumoController:
            default:
                var child = sumo.Find(vfxRootName);
                if (child == null)
                {
                    var go = new GameObject(vfxRootName);
                    go.layer = vfxLayer;
                    child = go.transform;
                    child.SetParent(sumo, false);
                    child.localPosition = Vector3.zero;
                    child.localRotation = Quaternion.identity;
                    child.localScale = Vector3.one;
                }
                else
                {
                    child.localScale = Vector3.one;
                }
                return child;
        }
    }

    private float RandomRange(float a, float b) => (float)(a + (b - a) * _rng.NextDouble());

    private void EmitPixelBurst(Transform parent, Vector2 origin, Vector2 dirBias, PixelBurstSettings s, Color? overrideColor = null)
    {
        dirBias = dirBias.sqrMagnitude < 1e-6f ? Vector2.up : dirBias.normalized;
        float half = s.SpreadDeg * 0.5f;

        for (int i = 0; i < s.Count; i++)
        {
            float ang = RandomRange(-half, half);
            Vector2 dir = Quaternion.Euler(0, 0, ang) * dirBias;
            float spd = s.Speed.Sample(_rng);
            Vector2 vel = dir * spd;

            float sz = s.Size.Sample(_rng);
            float lt = s.Lifetime.Sample(_rng);
            Color c = overrideColor ?? (s.Color != null ? s.Color.Evaluate(RandomRange(0f, 1f)) : Color.white);

            var px = _pixelPool.Spawn(origin);
            if (parent != null) px.transform.SetParent(parent, true);
            px.Initialize(size: sz, color: c, lifetime: lt, startVelocity: vel, gravityY: s.GravityY, sortingOrder: vfxSortingOrder, onDone: _pixelPool.Despawn);
        }
    }

}
