using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class SFXManager : MonoBehaviour
{
    public static SFXManager Instance { get; private set; }

    [System.Serializable]
    public class SFXBank
    {
        [Tooltip("Category key, e.g., 'collision', 'actions', 'ui'")]
        public string key = "default";
        [Tooltip("Clips to pick randomly from")]
        public List<AudioClip> clips = new List<AudioClip>();

        [Header("Per-Bank Tweaks")]
        [Range(0f, 1f)] public float volume = 1f;
        [Tooltip("Random pitch range (x=min, y=max). 1 = normal.")]
        public Vector2 pitchRange = new Vector2(0.95f, 1.05f);
    }

    [Header("Banks")]
    [SerializeField]
    private List<SFXBank> banks = new List<SFXBank>()
    {
        new SFXBank(){ key="collision" },
        new SFXBank(){ key="actions"   },
        new SFXBank(){ key="ui"        }
    };

    [Header("General")]
    [SerializeField, Range(1, 32)] private int poolSize = 12;
    [SerializeField] private bool dontDestroyOnLoad = true;

    // Internals
    private readonly Dictionary<string, SFXBank> bankMap = new Dictionary<string, SFXBank>();
    private readonly List<AudioSource> pool = new List<AudioSource>();
    private int nextIdx = 0;
    private Transform poolRoot;          // parent for pooled sources
    private bool isQuitting = false;     // avoid recreating on quit

    private bool isActive => Instance != null && gameObject != null && gameObject.activeSelf;

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
        if (dontDestroyOnLoad) DontDestroyOnLoad(gameObject);

        bankMap.Clear();
        foreach (var b in banks)
        {
            if (string.IsNullOrWhiteSpace(b.key)) continue;
            bankMap[b.key] = b;
        }

        // Create a dedicated child to host all pooled 2D sources
        var rootGo = new GameObject("SFX_2D_Pool");
        rootGo.transform.SetParent(transform, false);
        poolRoot = rootGo.transform;

        // Build initial pool
        pool.Clear();
        for (int i = 0; i < poolSize; i++)
            pool.Add(Create2DSource());
        nextIdx = 0;
    }

    void OnApplicationQuit() => isQuitting = true;

    // ---------- Public API ----------

    /// <summary>Play a random clip from a bank as 2D (non-positional).</summary>
    public void Play2D(string key)
    {
        if (!isActive) return;

        var (clip, vol, pitch) = Pick(key);
        if (clip == null) return;

        var src = GetNext2DSource();
        if (!src) return; // destroyed during shutdown or couldn't recreate

        src.pitch = pitch;
        src.volume = vol; // kept for consistency; OneShot uses its own volume param
        src.PlayOneShot(clip, vol);
    }

    public (AudioSource, float vol, float pitch) GetAudioSource(string key)
    {
        if (!isActive) return (null, -1, -1);

        var (clip, vol, pitch) = Pick(key);
        if (clip == null) return (null, -1, -1);


        var src = GetNext2DSource();
        if (!src) return (null, -1, -1); // destroyed during shutdown or couldn't recreate

        src.clip = clip;
        src.volume = vol;
        src.pitch = pitch;

        return (src, vol, pitch);
    }

    /// <summary>Play a random clip from a bank at a position (3D).</summary>
    public void Play3D(string key, Vector3 position, float spatialBlend = 1f, float maxDistance = 25f, float minDistance = 1f)
    {
        if (!isActive) return;

        var (clip, vol, pitch) = Pick(key);
        if (clip == null) return;

        // Lightweight temp AudioSource (auto-destroy)
        var go = new GameObject($"SFX3D_{key}");
        go.transform.position = position;

        var src = go.AddComponent<AudioSource>();
        src.spatialBlend = Mathf.Clamp01(spatialBlend);
        src.rolloffMode = AudioRolloffMode.Linear;
        src.minDistance = Mathf.Max(0.01f, minDistance);
        src.maxDistance = Mathf.Max(src.minDistance + 0.01f, maxDistance);
        src.playOnAwake = false;
        src.loop = false;
        src.pitch = pitch;
        src.volume = vol;
        src.clip = clip;

        src.Play();
        Destroy(go, clip.length / Mathf.Max(0.01f, src.pitch) + 0.1f);
    }

    /// <summary>Manually add/replace a bank at runtime.</summary>
    public void SetBank(SFXBank bank)
    {
        if (!isActive) return;
        
        if (bank == null || string.IsNullOrWhiteSpace(bank.key)) return;
        bankMap[bank.key] = bank;
    }

    // ---------- Helpers ----------

    private (AudioClip clip, float vol, float pitch) Pick(string key)
    {
        if (!bankMap.TryGetValue(key, out var bank) || bank.clips == null || bank.clips.Count == 0)
        {
            Debug.LogWarning($"[SFXManager] Bank '{key}' missing or empty.");
            return (null, 0f, 1f);
        }
        var clip = bank.clips[Random.Range(0, bank.clips.Count)];
        var pitch = Random.Range(bank.pitchRange.x, bank.pitchRange.y);
        var vol = Mathf.Clamp01(bank.volume);
        return (clip, vol, pitch);
    }

    // --- Pooling ---

    private AudioSource GetNext2DSource()
    {
        CompactPool(); // remove destroyed & ensure at least one

        if (pool.Count == 0) return null; // e.g., quitting

        // Round-robin
        var idx = nextIdx;
        nextIdx = (nextIdx + 1) % pool.Count;

        var src = pool[idx];

        // Defensive: if Unity "fake-null" or disabled, replace this slot
        if (!src || !src.gameObject || !src.enabled)
        {
            var replacement = Create2DSource();
            if (replacement)
            {
                src.Stop();
                pool[idx] = replacement;
                src = replacement;
            }
            else
            {
                // if cannot recreate (quitting), just bail
                return null;
            }
        }

        return src;
    }

    private void CompactPool()
    {
        // Remove any destroyed AudioSources (Unity fake-null check)
        for (int i = pool.Count - 1; i >= 0; i--)
        {
            if (!pool[i]) pool.RemoveAt(i);
        }

        if (pool.Count == 0 && !isQuitting)
        {
            // Rebuild at least one so Play2D keeps working
            pool.Add(Create2DSource());
            nextIdx = 0;
        }
        else if (nextIdx >= pool.Count && pool.Count > 0)
        {
            nextIdx = 0;
        }

        // Top up to poolSize if something got destroyed
        while (pool.Count < poolSize && !isQuitting)
            pool.Add(Create2DSource());
    }

    private AudioSource Create2DSource()
    {
        if (isQuitting) return null;

        var go = new GameObject("SFX_2D_Source");
        go.transform.SetParent(poolRoot, false);

        var src = go.AddComponent<AudioSource>();
        src.playOnAwake = false;
        src.loop = false;
        src.spatialBlend = 0f;   // 2D
        src.dopplerLevel = 0f;

        return src;
    }
}
