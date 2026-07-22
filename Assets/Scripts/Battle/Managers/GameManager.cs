

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using SumoCore;
using SumoServices;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    private static GameManager instance;
    public static GameManager Instance
    {
        get
        {
            if (instance == null)
            {
                GameObject go = new("GameManager");
                instance = go.AddComponent<GameManager>();
                DontDestroyOnLoad(go);
            }
            return instance;
        }
    }

    // Temporary
    public PlayerProfile Left;
    public PlayerProfile Right;

    // Edit Robot Mode
    public string EditingID;
    public bool ShowReplay = false;

    Scene savedBattleScene;

    void Awake()
    {
        if (instance != null && instance != this)
        {
            Destroy(gameObject);
            return;
        }

        instance = this;
        DontDestroyOnLoad(gameObject);
    }

    // Kept off Awake so scene startup never blocks on (future) network I/O.
    public async Task<ServiceResult<PlayerAccount>> StartSessionAsync()
    {
        var result = await GameServices.StartSessionAsync();
        if (!result.Success)
            Logger.Error($"[GameManager] Session start failed: {result.Error}");
        return result;
    }

    void OnEnable()
    {
        // Persistent local profiles so leaderboard identity survives restarts.
        Left = PlayerProfile.LoadOrCreate("Sumobot.Profile.Left", "Player1");
        Right = PlayerProfile.LoadOrCreate("Sumobot.Profile.Right", "Player2");
    }

    /// <summary>
    /// Call this once after a successful login. It switches the local player's
    /// identity from the anonymous device GUID to the server account, persists
    /// it, and carries any anonymous leaderboard ratings over to the account.
    /// </summary>
    public void ApplyAccount(string accountId, string accountName)
    {
        if (string.IsNullOrWhiteSpace(accountId))
            return;
        if (accountId.StartsWith("bot:"))
        {
            Logger.Error("[GameManager] Account IDs must not use the reserved 'bot:' prefix.");
            return;
        }

        string oldId = Left.ID;
        Left.ID = accountId;
        if (!string.IsNullOrWhiteSpace(accountName))
            Left.Name = accountName.Trim();

        PlayerPrefs.SetString("Sumobot.Profile.Left.ID", Left.ID);
        PlayerPrefs.SetString("Sumobot.Profile.Left.Name", Left.Name);
        PlayerPrefs.Save();

        if (oldId != accountId)
            GameServices.Leaderboard.ReassignProfile(oldId, accountId, Left.Name);
    }

    public void Battle_LoadCostumeScene(string id)
    {
        EditingID = id;
        SceneManager.LoadScene("BotCreator");
    }

    public void Battle_ShowReplay()
    {
        ShowReplay = true;
        SceneManager.LoadScene("Replay");
    }
    public void Replay_BackToBattle()
    {
        SceneManager.LoadScene("Battle");
        ShowReplay = false;
    }

    public void BotCreator_SaveAndBack()
    {
        SceneManager.LoadScene("Battle");
    }

    public PlayerProfile GetProfileById()
    {
        if (Left.ID == EditingID)
        {
            return Left;
        }
        else if (Right.ID == EditingID)
        {
            return Right;
        }
        return null;
    }
}

public class PlayerProfile
{
    public string ID;
    public string Name;
    public Dictionary<SumoPart, Sprite> Parts = new()
        {
            {SumoPart.Wheel, null},
            {SumoPart.Eye, null},
            {SumoPart.Accessory, null},
            {SumoPart.Body, null},
        };

    /// <summary>
    /// Per-part tint applied alongside <see cref="Parts"/>. Lets a color variant that shares
    /// one base sprite (e.g. Accessory color skins) render correctly. White = no tint.
    /// FaceSide is intentionally absent — its color is owned by SumoCostume.UpdateSideColor
    /// (red/green side marker) and must not be overwritten by a skin.
    /// </summary>
    public Dictionary<SumoPart, Color> PartTints = new()
        {
            {SumoPart.Wheel, Color.white},
            {SumoPart.Eye, Color.white},
            {SumoPart.Accessory, Color.white},
            {SumoPart.Body, Color.white},
        };

    public SumoCostume CurrentCostume;

    public static PlayerProfile CreateProfile(string name)
    {
        PlayerProfile profile = new()
        {
            Name = name,
            ID = Guid.NewGuid().ToString()
        };
        profile.PrepareParts();
        return profile;
    }

    /// <summary>
    /// Loads a profile persisted in PlayerPrefs under <paramref name="prefsKey"/>,
    /// creating (and saving) a new one on first run. Keeps the profile ID stable
    /// across sessions, which the leaderboard relies on.
    /// </summary>
    public static PlayerProfile LoadOrCreate(string prefsKey, string defaultName)
    {
        string idKey = $"{prefsKey}.ID";
        string nameKey = $"{prefsKey}.Name";

        string id = PlayerPrefs.GetString(idKey, string.Empty);
        string name = PlayerPrefs.GetString(nameKey, defaultName);

        if (string.IsNullOrEmpty(id))
        {
            id = Guid.NewGuid().ToString();
            PlayerPrefs.SetString(idKey, id);
            PlayerPrefs.SetString(nameKey, name);
            PlayerPrefs.Save();
        }

        PlayerProfile profile = new()
        {
            Name = name,
            ID = id
        };
        profile.PrepareParts();
        return profile;
    }

    /// <summary>Renames the profile and persists the new name.</summary>
    public void Rename(string prefsKey, string newName)
    {
        if (string.IsNullOrWhiteSpace(newName))
            return;
        Name = newName.Trim();
        PlayerPrefs.SetString($"{prefsKey}.Name", Name);
        PlayerPrefs.Save();
    }

    public void SetCostume(SumoCostume objectCostume)
    {
        ApplyEquippedForLocalPlayer();
        CurrentCostume = objectCostume;
        CurrentCostume.UpdateSideColor();
        CurrentCostume.AttachObject(Parts, PartTints);
    }

    /// <summary>
    /// Overlays the logged-in player's equipped Catalog skins (PlayerData.EquippedBySlot)
    /// onto <see cref="Parts"/>/<see cref="PartTints"/> for the equippable slots. The bridge
    /// from a Catalog skin id to a costume sprite is <see cref="SkinItem.PartSprite"/>.
    ///
    /// Scope guard: this only applies to the profile whose ID matches the loaded PlayerData
    /// (i.e. the local logged-in player — see GameManager.ApplyAccount). For the opponent, an
    /// anonymous session, or when services aren't ready, it no-ops and the legacy default
    /// parts (PrepareParts / PartSwitcher) stand — so the two costume systems coexist untouched.
    /// Equippable slots: Wheel, Accessory, Body. FaceSide is never equipped (side marker).
    /// </summary>
    private void ApplyEquippedForLocalPlayer()
    {
        var data = GameServices.PlayerData?.Current;
        if (data == null || data.PlayerId != ID)
            return; // battle-scoped guard: only the local player's profile gets this treatment

        EquippedCostumeResolver.ApplyEquipped(Parts, PartTints);
    }

    public void PrepareParts()
    {
        Parts.ToList().ForEach((e) =>
        {
            // Default sprites are numbered (Wheel_1, Eye_1, Accessory_1), but the body ships
            // as a single un-numbered Body.png — variety comes from IconColor tint, not extra sprites.
            string spriteName = e.Key == SumoPart.Body ? "Body" : $"{e.Key}_1";
            Parts[e.Key] = LoadSprite(spriteName);
        });
    }

    private Sprite LoadSprite(string path)
    {
        var basePath = "Sprites/Character";
        var prefab = Resources.Load<Sprite>($"{basePath}/{path}");
        if (prefab == null)
        {
            Logger.Error($"Sprite not found at Resources/{path}");
            return null;
        }
        return prefab;
    }
}

public static class SceneUtils
{
    public static void SetSceneActive(string sceneName, bool active)
    {
        Scene scene = SceneManager.GetSceneByName(sceneName);
        if (!scene.isLoaded) return;

        foreach (GameObject rootObj in scene.GetRootGameObjects())
        {
            rootObj.SetActive(active);
        }
    }
}