using System.Collections.Generic;
using UnityEngine;

public class PixelParticle : MonoBehaviour
{
    private float _life, _age, _gravityY;
    private Vector2 _vel;
    private SpriteRenderer _sr;
    private System.Action<PixelParticle> _onDone;

    // One tiny cached 2x2 white sprite (point-filtered) for all particles
    private static Sprite _square;
    private static Sprite GetSquareSprite()
    {
        if (_square != null) return _square;
        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
        var cols = new Color[] { Color.white, Color.white, Color.white, Color.white };
        tex.SetPixels(cols);
        tex.Apply();
        _square = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(0.5f, 0.5f), 100f, 0, SpriteMeshType.FullRect);
        return _square;
    }

    public void Initialize(float size, Color color, float lifetime, Vector2 startVelocity, float gravityY, int sortingOrder, System.Action<PixelParticle> onDone)
    {
        if (_sr == null)
        {
            _sr = gameObject.AddComponent<SpriteRenderer>();
            _sr.sprite = GetSquareSprite();
            _sr.sortingLayerName = "Foreground";
        }
        _sr.sortingOrder = sortingOrder;
        _sr.color = color;
        transform.localScale = new Vector3(size, size, 1f);

        _life = Mathf.Max(0.01f, lifetime);
        _age = 0f;
        _gravityY = gravityY;
        _vel = startVelocity;
        _onDone = onDone;

        gameObject.SetActive(true);
        enabled = true;
    }

    private void Update()
    {
        float dt = Time.deltaTime;
        _age += dt;

        if (_age >= _life)
        {
            _onDone?.Invoke(this);
            return;
        }

        // simple motion
        _vel.y += _gravityY * dt;
        transform.position += (Vector3)(_vel * dt);

        // fade alpha over lifetime
        if (_sr != null)
        {
            var c = _sr.color;
            c.a = Mathf.Lerp(1f, 0f, _age / _life);
            _sr.color = c;
        }
    }
}

public class PixelParticlePool
{
    private readonly Stack<PixelParticle> _pool = new Stack<PixelParticle>();
    private readonly int _capacity;
    private readonly int _layer;
    private readonly int _sortingOrder;

    public PixelParticlePool(int capacity, int vfxLayer, int sortingOrder)
    {
        _capacity = Mathf.Max(8, capacity);
        _layer = vfxLayer;
        _sortingOrder = sortingOrder;
    }

    public PixelParticle Spawn(Vector2 worldPos)
    {
        PixelParticle p = (_pool.Count > 0) ? _pool.Pop() : Create();
        var tr = p.transform;
        tr.position = worldPos;
        p.gameObject.layer = _layer;
        return p;
    }

    public void Despawn(PixelParticle p)
    {
        if (p == null) return;
        if (_pool.Count < _capacity)
        {
            p.gameObject.SetActive(false);
            p.enabled = false;
            _pool.Push(p);
        }
        else
        {
            Object.Destroy(p.gameObject);
        }
    }

    private PixelParticle Create()
    {
        var go = new GameObject("VFX_Pixel");
        var pp = go.AddComponent<PixelParticle>();
        go.SetActive(false);
        return pp;
    }
}