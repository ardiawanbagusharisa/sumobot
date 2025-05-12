using BattleLoop;
using CoreSumoRobot;
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

    public void PrepareInput(SumoRobotController controller)
    {
        GameObject selectedInputObject = null;
        // Assigning UI Object to players
        if (controller.Side == PlayerSide.Left)
        {
            switch (BattleManager.Instance.BattleInputType)
            {
                case BattleInputType.UI:
                    LeftButton.SetActive(true);
                    var skillText = LeftButton.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                    BattleUIManager.Instance.LeftSpecialSkill = skillText;
                    selectedInputObject = LeftButton;

                    LeftLiveCommand.SetActive(false);
                    break;
                case BattleInputType.LiveCommand:
                    LeftLiveCommand.SetActive(true);
                    selectedInputObject = LeftLiveCommand;

                    LeftButton.SetActive(false);
                    break;
            }
        }
        else
        {
            switch (BattleManager.Instance.BattleInputType)
            {
                case BattleInputType.UI:
                    RightButton.SetActive(true);
                    var skillText = RightButton.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                    BattleUIManager.Instance.RightSpecialSkill = skillText;
                    selectedInputObject = RightButton;

                    RightLiveCommand.SetActive(false);
                    break;
                case BattleInputType.LiveCommand:
                    RightLiveCommand.SetActive(true);
                    selectedInputObject = RightLiveCommand;

                    RightButton.SetActive(false);
                    break;
            }
        }

        if (selectedInputObject == null)
        {
            throw new System.Exception("One of [BattleInputType]'s object must be used");
        }

        // Declare that Robot driven by an input provider
        InputProvider inputProvider = selectedInputObject.GetComponent<InputProvider>();
        inputProvider.SkillType = controller.Skill.Type;
        controller.InputProvider = inputProvider;

        // Additional initialization
        switch (BattleManager.Instance.BattleInputType)
        {
            case BattleInputType.UI:
                break;
            case BattleInputType.LiveCommand:
                LeftLiveCommand.GetComponent<LiveCommandInput>().Init(controller);
                RightLiveCommand.GetComponent<LiveCommandInput>().Init(controller);
                break;
        }
    }
}