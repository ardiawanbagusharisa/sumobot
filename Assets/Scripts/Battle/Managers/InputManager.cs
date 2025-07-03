using System;
using SumoBot;
using SumoCore;
using SumoInput;
using Unity.VisualScripting;
using UnityEngine;

namespace SumoManager
{
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
                    // UIButtonsObject.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                    selectedInputObject = UIButtonsObject;
                    liveCommandObject.SetActive(false);
                    break;
            }

            SumoAPI api = CreateAPI(controller.Side);
            InputProvider inputProvider;
            InputType battleInputType = BattleManager.Instance.BattleInputType;

            if (battleInputType == InputType.Script)
            {
                if (BattleManager.Instance.Bot.IsEnable)
                {
                    InputProvider scriptInputProvider = controller.AddComponent<InputProvider>();
                    scriptInputProvider.PlayerSide = controller.Side;
                    scriptInputProvider.IncludeKeyboard = false;
                    inputProvider = scriptInputProvider;
                }
                else
                    throw new Exception($"Battle with [{battleInputType}] should provide InputProvider");
            }
            else
            {
                if (selectedInputObject == null)
                    throw new Exception($"One of [{battleInputType}]'s object must be used");

                inputProvider = selectedInputObject.GetComponent<InputProvider>();
            }

            inputProvider.SkillType = controller.Skill.Type;
            controller.InputProvider = inputProvider;

            // Additional initialization
            switch (BattleManager.Instance.BattleInputType)
            {
                case InputType.Script:
                    SetupBots(controller.Side, inputProvider, api);
                    break;
                case InputType.UI:
                    // Enable for test-only
                    SetupBots(controller.Side, inputProvider, api);
                    break;
                case InputType.LiveCommand:
                    // Enable for test-only
                    SetupBots(controller.Side, inputProvider, api);
                    liveCommandObject.GetComponent<CommandSystem>().InitCommandSystem(api);
                    break;
            }
        }

        private SumoAPI CreateAPI(PlayerSide side)
        {
            SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
            SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;

            SumoAPI api;

            if (side == PlayerSide.Left)
            {
                api = new SumoAPI(leftPlayer, rightPlayer);
            }
            else
            {
                api = new SumoAPI(rightPlayer, leftPlayer);
            }
            return api;
        }

        private void SetupBots(PlayerSide side, InputProvider provider, SumoAPI api)
        {
            if (!BattleManager.Instance.Bot.IsEnable) return;

            SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
            SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;

            if (leftPlayer.Bot != null && side == PlayerSide.Left)
            {
                leftPlayer.Bot.SetProvider(provider);
                leftPlayer.Bot.OnBotInit(side, api);
                leftPlayer.Actions[SumoController.OnPlayerBounce].Subscribe(leftPlayer.Bot.OnBotCollision);
            }
            else if (rightPlayer.Bot != null && side == PlayerSide.Right)
            {
                rightPlayer.Bot.SetProvider(provider);
                rightPlayer.Bot.OnBotInit(side, api);
                rightPlayer.Actions[SumoController.OnPlayerBounce].Subscribe(rightPlayer.Bot.OnBotCollision);
            }
        }
        #endregion
    }
}