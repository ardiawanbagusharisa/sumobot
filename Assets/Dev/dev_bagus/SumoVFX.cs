using UnityEngine;
using System;

[DisallowMultipleComponent]
public class SumoVFX : MonoBehaviour
{
    struct SparkRuntime
    {
        public float size;
        public float speed;
        public float life;
        public int count;
        public Color startColor;
        public Color endColor;
        public float gravity;
    }

    [Header("Anchors")]
    public Transform vfxRoot;
    public Transform leftTrailAnchor;
    public Transform rightTrailAnchor;

    [Header("Sorting Layer")]
    public string sortingLayerName = "Default";
    public int sortingOrder = 10;
    public int vfxLayer = 0;

    [Header("Trail Settings")]
    public TrailSettings accelerationTrail = TrailSettings.DefaultAccelerate();
    public TrailSettings turnTrail = TrailSettings.DefaultTurn();

    [Header("Smoke Settings")]
    public SmokeSettings dashSmoke = SmokeSettings.Dash();
    public SmokeSettings boostSmoke = SmokeSettings.Boost();

    [Header("Spark Settings")]
    public CollisionSparkSettings collisionSpark = CollisionSparkSettings.Default();

    TrailRenderer _leftTR, _rightTR;

    void Reset()
    {
        sortingLayerName = "Default";
        sortingOrder = 10;
    }

    void Awake()
    {
        EnsureVFXRoot();
        EnsureTrail(ref leftTrailAnchor, ref _leftTR, "LeftTrail", accelerationTrail);
        EnsureTrail(ref rightTrailAnchor, ref _rightTR, "RightTrail", accelerationTrail);
    }

    void EnsureVFXRoot()
    {
        if (!vfxRoot)
        {
            var found = transform.Find("VFX");
            if (found) 
                vfxRoot = found;
            else
            {
                var go = new GameObject("VFX");
                vfxRoot = go.transform;
                vfxRoot.SetParent(transform, false);
                vfxRoot.localPosition = Vector3.zero;
            }
        }
        vfxRoot.gameObject.layer = vfxLayer;
    }

    void EnsureTrail(ref Transform anchor, ref TrailRenderer tr, string name, TrailSettings settings)
    {
        if (!anchor)
        {
            var found = vfxRoot.Find(name);
            if (found) 
                anchor = found;
            else
            {
                var go = new GameObject(name);
                anchor = go.transform;
                anchor.SetParent(vfxRoot, false);
                anchor.localPosition = Vector3.zero;
            }
        }
        anchor.gameObject.layer = vfxLayer;

        tr = anchor.GetComponent<TrailRenderer>();
        if (!tr) 
            tr = anchor.gameObject.AddComponent<TrailRenderer>();

        ApplyTrailSettings(tr, settings);
    }

    static void ApplyTrailSettings(TrailRenderer tr, TrailSettings s)
    {
        tr.time = s.lifetime;
        tr.minVertexDistance = s.minVertexDistance;
        tr.emitting = false;
        tr.numCapVertices = 2;
        tr.numCornerVertices = 2;
        tr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        tr.receiveShadows = false;
        tr.generateLightingData = false;
        tr.alignment = LineAlignment.View;

        var mat = tr.sharedMaterial;
        if (mat == null)
        {
            Shader shader =
                Shader.Find("Sprites/Default") ??
                Shader.Find("Universal Render Pipeline/Unlit");
            mat = new Material(shader);
            tr.sharedMaterial = mat;
        }

        tr.widthCurve = s.WidthCurve;
        tr.colorGradient = s.Gradient;

        tr.sortingLayerName = string.IsNullOrEmpty(s.sortingLayerName) ? "Default" : s.sortingLayerName;
        tr.sortingOrder = s.sortingOrder;
    }

    void SetTrailStyle(bool turning)
    {
        var style = turning ? turnTrail : accelerationTrail;
        if (_leftTR) 
            ApplyTrailSettings(_leftTR, style);
        if (_rightTR) 
            ApplyTrailSettings(_rightTR, style);
    }

    // ------------------- Public API (call from SumoController) -------------------
    public void SetAccelerationTrail(bool on)
    {
        SetTrailStyle(false);
        if (_leftTR) 
            _leftTR.emitting = on;
        if (_rightTR) 
            _rightTR.emitting = on;
    }

    public void SetTurning(bool on)
    {
        SetTrailStyle(on);
        if (_leftTR) 
            _leftTR.emitting = on || _leftTR.emitting;
        if (_rightTR) 
            _rightTR.emitting = on || _rightTR.emitting;
    }

    public void DashSmoke() => SpawnSmoke(dashSmoke);

    public void BoostSmoke() => SpawnSmoke(boostSmoke);

    public void CollisionSpark(float relativeSpeed)
    {
        float t = Mathf.InverseLerp(collisionSpark.minSpeed, collisionSpark.maxSpeed, relativeSpeed);
        var cfg = new SparkRuntime
        {
            size = Mathf.Lerp(collisionSpark.sizeMin, collisionSpark.sizeMax, t),
            speed = Mathf.Lerp(collisionSpark.speedMin, collisionSpark.speedMax, t),
            life = Mathf.Lerp(collisionSpark.lifeMin, collisionSpark.lifeMax, t),
            count = Mathf.RoundToInt(Mathf.Lerp(collisionSpark.countMin, collisionSpark.countMax, t)),
            startColor = collisionSpark.startColor,
            endColor = collisionSpark.endColor,
            gravity = collisionSpark.gravity
        };
        SpawnSparks(cfg);
    }

    public void CollisionSparkAt(Vector2 worldPos, float relativeSpeed)
    {
        float t = Mathf.InverseLerp(collisionSpark.minSpeed, collisionSpark.maxSpeed, relativeSpeed);

        var cfg = new SparkRuntime
        {
            size = Mathf.Lerp(collisionSpark.sizeMin, collisionSpark.sizeMax, t),
            speed = Mathf.Lerp(collisionSpark.speedMin, collisionSpark.speedMax, t),
            life = Mathf.Lerp(collisionSpark.lifeMin, collisionSpark.lifeMax, t),
            count = Mathf.RoundToInt(Mathf.Lerp(collisionSpark.countMin, collisionSpark.countMax, t)),
            startColor = collisionSpark.startColor,
            endColor = collisionSpark.endColor,
            gravity = collisionSpark.gravity
        };

        var go = new GameObject("SparkPS@Contact");
        go.layer = vfxLayer;

        // Parent under vfxRoot but keep world position
        var tr = go.transform;
        tr.SetParent(vfxRoot, true);
        tr.position = worldPos;

        var ps = go.AddComponent<ParticleSystem>();
        var main = ps.main;
        var emission = ps.emission;
        var shape = ps.shape;
        var colOverLife = ps.colorOverLifetime;
        var sizeOverLife = ps.sizeOverLifetime;
        var renderer = ps.GetComponent<ParticleSystemRenderer>();

        EnsureParticleMaterial(renderer, sortingLayerName, sortingOrder);
        renderer.renderMode = ParticleSystemRenderMode.Stretch;
        renderer.lengthScale = 2.0f;

        main.duration = 1.0f;
        main.loop = false;
        main.startLifetime = cfg.life;
        main.startSpeed = cfg.speed;
        main.startSize = cfg.size;
        main.startColor = cfg.startColor;
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.gravityModifier = cfg.gravity;

        emission.rateOverTime = 0f;
        emission.SetBursts(new ParticleSystem.Burst[] {
            new ParticleSystem.Burst(0f, (short)Mathf.Max(1, cfg.count))
        });

        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Cone;
        shape.angle = 22f;
        shape.radius = 0.02f;

        colOverLife.enabled = true;
        var grad = new Gradient();
        grad.SetKeys(
            new[] {
            new GradientColorKey(cfg.startColor, 0f),
            new GradientColorKey(cfg.endColor,   1f),
            },
            new[] {
            new GradientAlphaKey(cfg.startColor.a, 0f),
            new GradientAlphaKey(cfg.endColor.a,   1f),
            }
        );
        colOverLife.color = new ParticleSystem.MinMaxGradient(grad);

        sizeOverLife.enabled = true;
        sizeOverLife.size = new ParticleSystem.MinMaxCurve(1f, new AnimationCurve(
            new Keyframe(0f, 1f),
            new Keyframe(1f, 0.5f)
        ));

        ps.Play();
        Destroy(go, main.startLifetime.constantMax + 0.5f);
    }

    void SpawnSmoke(SmokeSettings s)
    {
        // Create PS GameObject
        var go = new GameObject("SmokePS");
        go.layer = vfxLayer;
        var t = go.transform;
        t.SetParent(vfxRoot, false);
        t.localPosition = Vector3.zero;

        var ps = go.AddComponent<ParticleSystem>();
        var main = ps.main;
        var emission = ps.emission;
        var shape = ps.shape;
        var colOverLife = ps.colorOverLifetime;
        var sizeOverLife = ps.sizeOverLifetime;
        var velOverLife = ps.velocityOverLifetime;
        var renderer = ps.GetComponent<ParticleSystemRenderer>();

        // Material (simple alpha blended)
        EnsureParticleMaterial(renderer, sortingLayerName, sortingOrder);

        // Main
        main.duration = 2f;
        main.loop = false;
        main.startLifetime = s.startLifetime;
        main.startSpeed = s.startSpeed;
        main.startSize = s.startSize;
        main.startRotation = s.randomizeRotation
            ? new ParticleSystem.MinMaxCurve(0f, Mathf.Deg2Rad * 360f)
            : new ParticleSystem.MinMaxCurve(0f);
        main.startColor = s.startColor; // alpha controlled here
        main.simulationSpace = ParticleSystemSimulationSpace.World;

        // Emission: single burst
        emission.rateOverTime = 0f;
        emission.SetBursts(new ParticleSystem.Burst[]
        {
            new ParticleSystem.Burst(0f, (short)Mathf.Max(1, s.burstCount))
        });

        // Shape: small cone/sphere-ish
        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Cone;
        shape.angle = 10f;
        shape.radius = 0.05f;

        // Optional soft growth/fade
        sizeOverLife.enabled = s.enableSizeOverLifetime;
        if (s.enableSizeOverLifetime)
        {
            sizeOverLife.size = new ParticleSystem.MinMaxCurve(1f, new AnimationCurve(
                new Keyframe(0f, s.sizeOverLifetimeStart),
                new Keyframe(0.6f, s.sizeOverLifetimePeak),
                new Keyframe(1f, s.sizeOverLifetimeEnd)
            ));
        }

        // Color Over Lifetime gradient (e.g., fade to transparent / darker)
        colOverLife.enabled = s.useColorOverLifetime;
        if (s.useColorOverLifetime)
        {
            var grad = new Gradient();
            grad.SetKeys(
                new[]
                {
                    new GradientColorKey(s.startColor, 0f),
                    new GradientColorKey(s.colorOverLifetimeEnd, 1f),
                },
                new[]
                {
                    new GradientAlphaKey(s.startColor.a, 0f),
                    new GradientAlphaKey(s.colorOverLifetimeEnd.a, 1f),
                }
            );
            colOverLife.color = new ParticleSystem.MinMaxGradient(grad);
        }

        // Slight upward drift for smoke
        velOverLife.enabled = s.upwardDrift != 0f;
        if (s.upwardDrift != 0f)
        {
            velOverLife.space = ParticleSystemSimulationSpace.World;
            velOverLife.y = s.upwardDrift;
        }

        ps.Play();
        Destroy(go, main.startLifetime.constantMax + 0.75f);
    }

    void SpawnSparks(SparkRuntime r)
    {
        var go = new GameObject("SparkPS");
        go.layer = vfxLayer;
        var t = go.transform;
        t.SetParent(vfxRoot, false);
        t.localPosition = Vector3.zero;

        var ps = go.AddComponent<ParticleSystem>();
        var main = ps.main;
        var emission = ps.emission;
        var shape = ps.shape;
        var colOverLife = ps.colorOverLifetime;
        var sizeOverLife = ps.sizeOverLifetime;
        var forceOverLife = ps.forceOverLifetime;
        var renderer = ps.GetComponent<ParticleSystemRenderer>();

        // Renderer: use stretch billboard for streaky sparks
        EnsureParticleMaterial(renderer, sortingLayerName, sortingOrder);
        renderer.renderMode = ParticleSystemRenderMode.Stretch;
        renderer.lengthScale = 2.0f; // stretch trails feel

        // Main
        main.duration = 1.0f;
        main.loop = false;
        main.startLifetime = r.life;
        main.startSpeed = r.speed;
        main.startSize = r.size;
        main.startColor = r.startColor; // alpha here
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.gravityModifier = r.gravity;

        // Emission: single burst
        emission.rateOverTime = 0f;
        emission.SetBursts(new ParticleSystem.Burst[]
        {
            new ParticleSystem.Burst(0f, (short)Mathf.Max(1, r.count))
        });

        // Shape: small cone pointing forward
        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Cone;
        shape.angle = 22f;
        shape.radius = 0.02f;

        // Color Over Lifetime: fade to endColor (often transparent)
        colOverLife.enabled = true;
        var grad = new Gradient();
        grad.SetKeys(
            new[]
            {
                new GradientColorKey(r.startColor, 0f),
                new GradientColorKey(r.endColor, 1f),
            },
            new[]
            {
                new GradientAlphaKey(r.startColor.a, 0f),
                new GradientAlphaKey(r.endColor.a, 1f),
            }
        );
        colOverLife.color = new ParticleSystem.MinMaxGradient(grad);

        // Size Over Lifetime: shrink slightly
        sizeOverLife.enabled = true;
        sizeOverLife.size = new ParticleSystem.MinMaxCurve(1f, new AnimationCurve(
            new Keyframe(0f, 1f),
            new Keyframe(1f, 0.5f)
        ));

        // Force Over Lifetime: small outward push (optional)
        forceOverLife.enabled = false;

        ps.Play();
        Destroy(go, main.startLifetime.constantMax + 0.5f);
    }

    static void EnsureParticleMaterial(ParticleSystemRenderer r, string sortingLayer, int sortingOrder)
    {
        r.sortingLayerName = sortingLayer;
        r.sortingOrder = sortingOrder;

        if (r.sharedMaterial == null)
        {
            // Defaults for both Built-in and URP
            Shader sh =
                Shader.Find("Particles/Standard Unlit") ??
                Shader.Find("Sprites/Default") ??
                Shader.Find("Universal Render Pipeline/Unlit");

            var mat = new Material(sh);
            // If using "Particles/Standard Unlit", enable color/softness via keywords if needed
            r.sharedMaterial = mat;
        }
    }

    // ------------------- Settings Types -------------------

    [Serializable]
    public class TrailSettings
    {
        [Range(0.05f, 3f)] public float lifetime = 0.6f;
        [Range(0.01f, 1f)] public float minVertexDistance = 0.08f;
        [Range(0.01f, 2f)] public float startWidth = 0.25f;
        [Range(0.00f, 2f)] public float endWidth = 0.0f;

        [Range(0f, 1f)] public float startAlpha = 0.85f;
        [Range(0f, 1f)] public float endAlpha = 0.0f;

        public Color startColor = Color.white;
        public Color endColor = Color.white * 0.9f;

        public string sortingLayerName = "Default";
        public int sortingOrder = 50;

        public AnimationCurve WidthCurve => new AnimationCurve(
            new Keyframe(0, startWidth),
            new Keyframe(1, endWidth)
        );

        public Gradient Gradient
        {
            get
            {
                var g = new Gradient();
                g.SetKeys(
                    new[] {
                        new GradientColorKey(startColor, 0f),
                        new GradientColorKey(endColor,   1f)
                    },
                    new[] {
                        new GradientAlphaKey(startAlpha, 0f),
                        new GradientAlphaKey(endAlpha,   1f)
                    }
                );
                return g;
            }
        }

        public static TrailSettings DefaultAccelerate() => new TrailSettings
        {
            lifetime = 0.6f,
            minVertexDistance = 0.06f,
            startWidth = 0.22f,
            endWidth = 0.0f,
            startAlpha = 0.75f,
            endAlpha = 0.0f,
            startColor = new Color(1f, 1f, 1f, 1f),
            endColor = new Color(1f, 1f, 1f, 1f),
            sortingLayerName = "Default",
            sortingOrder = 50
        };

        public static TrailSettings DefaultTurn() => new TrailSettings
        {
            lifetime = 0.7f,
            minVertexDistance = 0.06f,
            startWidth = 0.22f,
            endWidth = 0.0f,
            startAlpha = 0.9f,
            endAlpha = 0.02f,
            startColor = new Color(0.4f, 0.4f, 0.4f, 1f),
            endColor = new Color(0.35f, 0.35f, 0.35f, 1f),
            sortingLayerName = "Default",
            sortingOrder = 50
        };
    }

    [Serializable]
    public class SmokeSettings
    {
        [Header("Burst")]
        [Range(1, 200)] public int burstCount = 20;

        [Header("Particle")]
        [Range(0.1f, 5f)] public float startSize = 1.4f;
        [Range(0f, 10f)] public float startSpeed = 1.5f;
        [Range(0.1f, 5f)] public float startLifetime = 0.9f;
        public bool randomizeRotation = true;
        public Color startColor = new Color(0.9f, 0.9f, 0.9f, 0.9f); // alpha here

        [Header("Over Lifetime")]
        public bool useColorOverLifetime = false;
        public Color colorOverLifetimeEnd = new Color(0.8f, 0.8f, 0.8f, 0.0f);
        public bool enableSizeOverLifetime = false;
        [Range(0.1f, 2f)] public float sizeOverLifetimeStart = 0.6f;
        [Range(0.1f, 2f)] public float sizeOverLifetimePeak = 1.2f;
        [Range(0.0f, 2f)] public float sizeOverLifetimeEnd = 0.9f;

        [Header("Physics")]
        [Tooltip("World-space upward drift (m/s). 0 = off.")]
        public float upwardDrift = 0.6f;

        public static SmokeSettings Dash() => new SmokeSettings
        {
            burstCount = 18,
            startSize = 1.1f,
            startSpeed = 1.2f,
            startLifetime = 0.7f,
            randomizeRotation = true,
            useColorOverLifetime = false,
            colorOverLifetimeEnd = new Color(0.9f, 0.9f, 0.9f, 0f),
            enableSizeOverLifetime = true,
            sizeOverLifetimeStart = 0.6f,
            sizeOverLifetimePeak = 1.1f,
            sizeOverLifetimeEnd = 0.9f,
            upwardDrift = 0.5f
        };

        public static SmokeSettings Boost() => new SmokeSettings
        {
            burstCount = 32,
            startSize = 1.8f,
            startSpeed = 2.2f,
            startLifetime = 1.1f,
            randomizeRotation = true,
            useColorOverLifetime = false,
            colorOverLifetimeEnd = new Color(0.9f, 0.9f, 0.9f, 0f),
            enableSizeOverLifetime = true,
            sizeOverLifetimeStart = 0.7f,
            sizeOverLifetimePeak = 1.3f,
            sizeOverLifetimeEnd = 0.9f,
            upwardDrift = 0.7f
        };
    }

    [Serializable]
    public class CollisionSparkSettings
    {
        [Header("Speed thresholds")]
        public float minSpeed = 1.5f;
        public float maxSpeed = 10f;

        [Header("Particle ranges")]
        public float sizeMin = 0.05f;
        public float sizeMax = 0.18f;
        public float speedMin = 2f;
        public float speedMax = 6f;
        public float lifeMin = 0.2f;
        public float lifeMax = 0.5f;
        public int countMin = 8;
        public int countMax = 28;

        [Header("Look & Feel")]
        public Color startColor = new Color(1f, 0.8f, 0.3f, 1f);   // alpha here
        public Color endColor = new Color(1f, 0.5f, 0.1f, 0f);   // fade out
        public float gravity = 1.5f;

        public static CollisionSparkSettings Default() => new CollisionSparkSettings();
    }
}
