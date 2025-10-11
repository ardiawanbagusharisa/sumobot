using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class BGMManager : MonoBehaviour
{
    public static BGMManager Instance { get; private set; }

    [Header("Playback List")]
    [SerializeField] private List<AudioClip> tracks = new();

    [Header("Options")]
    [SerializeField] private bool dontDestroyOnLoad = true;
    [SerializeField] private bool playOnStart = true;
    [SerializeField] private bool randomizeOrder = true;
    [SerializeField, Min(0f)] private float gapBetweenTracks = 0f;

    [Header("Audio")]
    [Range(0f, 1f)][SerializeField] private float volume = 1f;
    [Range(0.25f, 3f)][SerializeField] private float playbackSpeed = 1f;

    [Header("Fades")]
    [Tooltip("Enable crossfading between tracks. If off, track A fades out then B fades in.")]
    [SerializeField] private bool crossfade = true;
    [SerializeField, Min(0f)] private float fadeInDuration = 1.0f;
    [SerializeField, Min(0f)] private float fadeOutDuration = 1.0f;

    // Internals
    private AudioSource[] sources;   // dual sources for crossfade
    private int activeSource = 0;
    private Coroutine playLoopCo;
    private Coroutine fadeCoA;
    private Coroutine fadeCoB;
    private int lastIndex = -1;

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
        if (dontDestroyOnLoad) DontDestroyOnLoad(gameObject);

        // Create / configure 2 sources
        sources = new AudioSource[2];
        for (int i = 0; i < 2; i++)
        {
            var src = gameObject.AddComponent<AudioSource>();
            src.playOnAwake = false;
            src.loop = false;
            src.volume = 0f;
            src.pitch = playbackSpeed;
            src.spatialBlend = 0f; // 2D
            sources[i] = src;
        }
    }

    void Start()
    {
        if (playOnStart && tracks.Count > 0) Play();
    }

    void OnValidate()
    {
        volume = Mathf.Clamp01(volume);
        playbackSpeed = Mathf.Clamp(playbackSpeed, 0.25f, 3f);
        if (sources != null)
        {
            foreach (var s in sources) if (s) s.pitch = playbackSpeed;
        }
    }

    // ---------------- Public API ----------------

    public void Play()
    {
        if (tracks == null || tracks.Count == 0)
        {
            Debug.LogWarning("[BGMManager] No tracks assigned.");
            return;
        }
        if (playLoopCo == null) playLoopCo = StartCoroutine(PlayLoop());
        else Resume();
    }

    public void Pause()
    {
        foreach (var s in sources) if (s && s.isPlaying) s.Pause();
    }

    public void Resume()
    {
        foreach (var s in sources) if (s && !s.isPlaying) s.UnPause();
    }

    public void Stop(bool instant = false)
    {
        if (playLoopCo != null) { StopCoroutine(playLoopCo); playLoopCo = null; }
        if (instant)
        {
            foreach (var s in sources) if (s) { s.Stop(); s.volume = 0f; }
        }
        else
        {
            // Graceful fade out everything
            for (int i = 0; i < sources.Length; i++)
                StartFade(sources[i], sources[i].volume, 0f, fadeOutDuration);
        }
    }

    public void Skip()
    {
        if (playLoopCo != null)
        {
            // Interrupt loop to immediately transition to next
            StopCoroutine(playLoopCo);
            playLoopCo = StartCoroutine(PlayLoop(forceNext: true));
        }
        else
        {
            Play();
        }
    }

    public void SetVolume(float v)
    {
        volume = Mathf.Clamp01(v);
        // Apply target volume to currently active/next fades
        // (fade coroutines will use 'volume' as their target max)
        for (int i = 0; i < sources.Length; i++)
            if (sources[i] && !sources[i].isPlaying) sources[i].volume = 0f;
    }

    public void SetPlaybackSpeed(float speed)
    {
        playbackSpeed = Mathf.Clamp(speed, 0.25f, 3f);
        foreach (var s in sources) if (s) s.pitch = playbackSpeed;
    }

    public void SetTracks(List<AudioClip> newTracks, bool restart = true)
    {
        tracks = newTracks ?? new List<AudioClip>();
        lastIndex = -1;
        if (restart)
        {
            Stop(true);
            if (tracks.Count > 0) Play();
        }
    }

    public void FadeOut(float duration) => StartFade(GetActive(), GetActive().volume, 0f, Mathf.Max(0f, duration));
    public void FadeIn(float duration) => StartFade(GetActive(), GetActive().volume, volume, Mathf.Max(0f, duration));

    // ---------------- Core Loop ----------------

    private IEnumerator PlayLoop(bool forceNext = false)
    {
        while (true)
        {
            int nextIndex = GetNextTrackIndex();
            if (nextIndex < 0) yield break;

            var nextClip = tracks[nextIndex];

            if (crossfade)
            {
                // Crossfade: start next on the inactive source while fading out the active one
                int nextSource = 1 - activeSource;
                PrepareSource(sources[nextSource], nextClip, startVolume: 0f);
                sources[nextSource].Play();

                // Fade up next, fade down current (in parallel)
                StartFade(sources[nextSource], 0f, volume, fadeInDuration);
                StartFade(sources[activeSource], sources[activeSource].volume, 0f, fadeOutDuration);

                // Swap active after a small moment (let it start)
                activeSource = nextSource;

                // Wait until the newly playing clip ends
                yield return new WaitWhile(() => sources[activeSource].isPlaying);

                if (gapBetweenTracks > 0f) yield return new WaitForSeconds(gapBetweenTracks);
            }
            else
            {
                // Non-crossfade: fade out current (if any), then play next and fade in
                if (sources[activeSource].isPlaying)
                {
                    yield return FadeRoutineBlocking(sources[activeSource], sources[activeSource].volume, 0f, fadeOutDuration);
                    sources[activeSource].Stop();
                }

                PrepareSource(sources[activeSource], nextClip, startVolume: 0f);
                sources[activeSource].Play();

                // Fade in
                yield return FadeRoutineBlocking(sources[activeSource], 0f, volume, fadeInDuration);

                // Wait until end
                yield return new WaitWhile(() => sources[activeSource].isPlaying);

                if (gapBetweenTracks > 0f) yield return new WaitForSeconds(gapBetweenTracks);
            }

            lastIndex = nextIndex;

            if (forceNext)
            {
                forceNext = false;
                // Immediately continue loop to select a new track
            }
        }
    }

    private int GetNextTrackIndex()
    {
        if (tracks == null || tracks.Count == 0) return -1;
        if (!randomizeOrder) return (lastIndex + 1) % tracks.Count;
        if (tracks.Count == 1) return 0;

        int idx;
        do { idx = Random.Range(0, tracks.Count); } while (idx == lastIndex);
        return idx;
    }

    private void PrepareSource(AudioSource s, AudioClip clip, float startVolume)
    {
        s.clip = clip;
        s.volume = Mathf.Clamp01(startVolume);
        s.pitch = playbackSpeed;
        s.loop = false;
        s.spatialBlend = 0f;
    }

    private AudioSource GetActive() => sources[activeSource];

    // ---------------- Fading Helpers ----------------

    private void StartFade(AudioSource s, float from, float to, float dur)
    {
        if (s == null) return;
        // Kill old fade on this specific source (track by reference)
        if (s == sources[0] && fadeCoA != null) StopCoroutine(fadeCoA);
        if (s == sources[1] && fadeCoB != null) StopCoroutine(fadeCoB);

        var co = StartCoroutine(FadeRoutine(s, from, to, dur));
        if (s == sources[0]) fadeCoA = co;
        else fadeCoB = co;
    }

    private IEnumerator FadeRoutine(AudioSource s, float from, float to, float dur)
    {
        if (dur <= 0f) { s.volume = Mathf.Clamp01(to); yield break; }
        float t = 0f;
        s.volume = Mathf.Clamp01(from);
        while (t < dur)
        {
            t += Time.unscaledDeltaTime; // unaffected by timescale
            float a = Mathf.Clamp01(t / dur);
            s.volume = Mathf.Lerp(from, to, a);
            yield return null;
        }
        s.volume = Mathf.Clamp01(to);
    }

    private IEnumerator FadeRoutineBlocking(AudioSource s, float from, float to, float dur)
    {
        yield return FadeRoutine(s, from, to, dur);
    }
}
