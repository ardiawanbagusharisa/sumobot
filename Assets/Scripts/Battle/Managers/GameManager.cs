

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

        GameServices.Initialize();
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
            SumoLeaderboard.LeaderboardService.Instance.ReassignProfile(oldId, accountId, Left.Name);
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
        CurrentCostume = objectCostume;
        CurrentCostume.UpdateSideColor();
        CurrentCostume.AttachObject(Parts);
    }

    public void PrepareParts()
    {
        Parts.ToList().ForEach((e) =>
        {
            Parts[e.Key] = LoadSprite($"{e.Key}_1");
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