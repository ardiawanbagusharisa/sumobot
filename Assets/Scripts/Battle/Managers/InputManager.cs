using System;
using System.Linq;
using CoreSumo;
using Unity.VisualScripting;
using UnityEngine;

public class InputManager : MonoBehaviour
{
    public static InputManager Instance { get; private set; }

    #region Input properties
    public GameObject LeftButton;
    public GameObject RightButton;
    public GameObject LeftLiveCommand;
    public GameObject RightLiveCommand;
    #endregion

    #region Unity methods
    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }
    #endregion

    #region Input methods
    public void InitializeInput(SumoController controller)
    {
        GameObject selectedInputObject;

        GameObject liveCommandObject = controller.Side == PlayerSide.Left ? LeftLiveCommand : RightLiveCommand;
        GameObject UIButtonsObject = controller.Side == PlayerSide.Left ? LeftButton : RightButton;

        switch (BattleManager.Instance.BattleInputType)
        {
            case InputType.Script:
                liveCommandObject.SetActive(false);
                UIButtonsObject.SetActive(false);
                selectedInputObject = null;
                break;

            case InputType.LiveCommand:
                liveCommandObject.SetActive(true);
                selectedInputObject = liveCommandObject;
                UIButtonsObject.SetActive(false);
                break;

            // UI button & keyboard 
            default:
                UIButtonsObject.SetActive(true);
                UIButtonsObject.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                selectedInputObject = UIButtonsObject;
                liveCommandObject.SetActive(false);
                break;
        }

        InputProvider inputProvider;

        if (BattleManager.Instance.BattleInputType == InputType.Script)
        {
            if (BattleManager.Instance.Bot.IsEnable)
            {
                var scriptInputProvider = controller.AddComponent<InputProvider>();
                scriptInputProvider.PlayerSide = controller.Side;
                scriptInputProvider.IncludeKeyboard = false;
                inputProvider = scriptInputProvider;
            }
            else
            {
                throw new Exception("Battle with [InputType.Script] should provide InputProvider");
            }
        }
        else
        {
            if (selectedInputObject == null)
            {
                throw new Exception("One of [BattleInputType]'s object must be used");
            }
            inputProvider = selectedInputObject.GetComponent<InputProvider>();
        }

        inputProvider.SkillType = controller.Skill.Type;
        controller.InputProvider = inputProvider;

        // Might be called only when the BattleInputType is Script
        // For now, test it whatever on the input type is set
        SetBots(controller.Side, inputProvider);

        // Additional initialization
        switch (BattleManager.Instance.BattleInputType)
        {
            case InputType.Script:
                break;
            case InputType.UI:
                break;
            case InputType.LiveCommand:
                break;
        }
    }

    private void SetBots(PlayerSide side, InputProvider provider)
    {
        if (!BattleManager.Instance.Bot.IsEnable) return;

        var me = side == PlayerSide.Left ? BattleManager.Instance.Battle.LeftPlayer : BattleManager.Instance.Battle.RightPlayer;
        var enemy = side == PlayerSide.Left ? BattleManager.Instance.Battle.RightPlayer : BattleManager.Instance.Battle.LeftPlayer;

        Bot leftBot = BattleManager.Instance.Bot.Left;
        Bot rightBot = BattleManager.Instance.Bot.Right;
        if (leftBot != null && side == PlayerSide.Left)
        {
            leftBot.SetProvider(provider);
            leftBot.OnBotInit(side, new BotAPI(me, enemy.transform));
            BattleManager.Instance.Battle.LeftPlayer.OnPlayerBounce += leftBot.OnBotCollision;
        }
        else if (rightBot != null && side == PlayerSide.Right)
        {
            rightBot.SetProvider(provider);
            rightBot.OnBotInit(side, new BotAPI(me, enemy.transform));
            BattleManager.Instance.Battle.RightPlayer.OnPlayerBounce += rightBot.OnBotCollision;
        }
    }

    public void ResetCooldownButton()
    {
        if (BattleManager.Instance.BattleInputType == InputType.UI || BattleManager.Instance.BattleInputType == InputType.Keyboard)
        {
            LeftButton.GetComponent<ButtonInputHandler>().ResetCooldown();
            RightButton.GetComponent<ButtonInputHandler>().ResetCooldown();
        }
    }
    #endregion
}