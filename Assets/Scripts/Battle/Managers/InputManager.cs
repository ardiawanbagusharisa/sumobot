using System;
using System.Linq;
using CoreSumo;
using Unity.VisualScripting;
using UnityEngine;

public class InputManager : MonoBehaviour
{
    public static InputManager Instance { get; private set; }

    public GameObject LeftButton;
    public GameObject RightButton;

    public GameObject LeftLiveCommand;
    public GameObject RightLiveCommand;

    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;

    }

    public void PrepareInput(SumoController controller)
    {
        GameObject selectedInputObject = null;
        // Assigning UI Object to players
        if (controller.Side == PlayerSide.Left)
        {
            switch (BattleManager.Instance.BattleInputType)
            {

                case InputType.Script:
                    LeftLiveCommand.SetActive(false);
                    LeftButton.SetActive(false);
                    break;
                case InputType.LiveCommand:
                    LeftLiveCommand.SetActive(true);
                    selectedInputObject = LeftLiveCommand;

                    LeftButton.SetActive(false);
                    break;

                // Handle UI And Keyboard
                default:
                    LeftButton.SetActive(true);
                    LeftButton.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                    selectedInputObject = LeftButton;

                    LeftLiveCommand.SetActive(false);
                    break;

            }
        }
        else
        {
            switch (BattleManager.Instance.BattleInputType)
            {
                case InputType.Script:
                    RightLiveCommand.SetActive(false);
                    RightButton.SetActive(false);
                    break;
                case InputType.LiveCommand:
                    RightLiveCommand.SetActive(true);
                    selectedInputObject = RightLiveCommand;

                    RightButton.SetActive(false);
                    break;

                // Handle UI And Keyboard
                default:
                    RightButton.SetActive(true);
                    RightButton.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                    selectedInputObject = RightButton;

                    RightLiveCommand.SetActive(false);
                    break;
            }
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
                throw new Exception("One of [BattleInputType]'s object must be used");
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

        // Declare that Robot driven by an input provider
        inputProvider.SkillType = controller.Skill.Type;
        controller.InputProvider = inputProvider;

        // Might be called only when the BattleInputType is Script
        // For now, test it whatever on the input type is set
        AssignBotsIfExist(controller.Side, inputProvider);

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

    private void AssignBotsIfExist(PlayerSide side, InputProvider provider)
    {
        var me = side == PlayerSide.Left ? BattleManager.Instance.Battle.LeftPlayer : BattleManager.Instance.Battle.RightPlayer;
        var enemy = side == PlayerSide.Left ? BattleManager.Instance.Battle.RightPlayer : BattleManager.Instance.Battle.LeftPlayer;
        
        if (BattleManager.Instance.Bot.Left != null && side == PlayerSide.Left)
        {
            BattleManager.Instance.Bot.Left.SetProvider(provider);
            BattleManager.Instance.Bot.Left.OnBotInit(side, new BotAPI(me, enemy.transform));
            BattleManager.Instance.Battle.LeftPlayer.OnPlayerBounce += BattleManager.Instance.Bot.Left.OnBotCollision;
        }
        else if (BattleManager.Instance.Bot.Right != null && side == PlayerSide.Right)
        {
            BattleManager.Instance.Bot.Right.SetProvider(provider);
            BattleManager.Instance.Bot.Right.OnBotInit(side, new BotAPI(me, enemy.transform));
            BattleManager.Instance.Battle.RightPlayer.OnPlayerBounce += BattleManager.Instance.Bot.Right.OnBotCollision;
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
}