

using System;
using System.Collections.Generic;
using System.Linq;
using SumoCore;
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

    void OnEnable()
    {
        // Dummy
        Left = PlayerProfile.CreateProfile("Player1");
        Right = PlayerProfile.CreateProfile("Player2");
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
            Debug.LogError($"Sprite not found at Resources/{path}");
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