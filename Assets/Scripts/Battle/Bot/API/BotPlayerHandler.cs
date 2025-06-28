using System;
using System.Linq;
using CoreSumo;
using UnityEngine;

[Serializable]
public class BotPlayerHandler
{
    public bool IsEnable;
    public bool IsScriptable = false;
    public Bot Left;
    public Bot Right;

    public void Init(GameObject leftGameObject, GameObject rightGameObject)
    {
        if (!IsEnable) return;
        if (IsScriptable) return;

        Left = leftGameObject.GetComponent<SumoController>().Bot;
        Right = rightGameObject.GetComponent<SumoController>().Bot;
    }

    public void OnUpdate(float ElapsedTime)
    {
        if (!IsEnable) return;

        if (Left != null)
        {
            Left.ElapsedTime = ElapsedTime;
            if (Left.ElapsedTime >= Left.Interval)
            {
                Left.ElapsedTime = 0;
                Left.OnBotUpdate();
            }
        }

        if (Right != null)
        {
            Right.ElapsedTime = ElapsedTime;
            if (Right.ElapsedTime >= Right.Interval)
            {
                Right.ElapsedTime = 0;
                Right.OnBotUpdate();
            }
        }
    }
    public void OnBattleStateChanged(BattleState currState)
    {
        if (!IsEnable) return;

        if (Left != null)
        {
            Left.OnBattleStateChanged(currState);
        }
        if (Right != null)
        {
            Right.OnBattleStateChanged(currState);
        }
    }
}