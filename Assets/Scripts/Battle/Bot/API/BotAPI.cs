using CoreSumo;
using UnityEngine;

public class BotAPI
{
    // [Todo]: Should change the SumoController to exact attribute, giving SumoController directly might be risky
    // Need to configure asmdef (assembly scope)
    internal readonly SumoController myController;
    internal readonly Transform enemyTransform;

    public BotAPI(SumoController myController, Transform enemyTransform)
    {
        this.myController = myController;
        this.enemyTransform = enemyTransform;
    }

    public Transform MyTransform
    {
        get
        {
            return myController.transform;
        }
    }

    public Transform EnemyTransform
    {
        get
        {
            return enemyTransform;
        }
    }

    public Vector2 GetLastVelocity()
    {
        return myController.LastVelocity;
    }

    public SumoSkill GetSkillAttr()
    {
        return myController.Skill;
    }

    public bool CanExecute(ISumoAction action)
    {
        return Controller.InputProvider.CanExecute(action);
    }

    public SumoController Controller
    {
        get
        {
            return myController;
        }
    }
}