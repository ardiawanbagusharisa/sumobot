using System;

[Serializable]
public class BotPlayer
{
    public bool IsEnable;
    public Bot Left;
    public Bot Right;

    public void OnUpdate(float ElapsedTime)
    {
        if (Left != null)
        {
            Left.BotElapsed = ElapsedTime;
            if (Left.BotElapsed >= Left.Interval)
            {
                Left.BotElapsed = 0;
                Left.OnBotUpdate();
            }
        }

        if (Right != null)
        {
            Right.BotElapsed = ElapsedTime;
            if (Right.BotElapsed >= Right.Interval)
            {
                Right.BotElapsed = 0;
                Right.OnBotUpdate();
            }
        }
    }
    public void OnBattleStateChanged(BattleState currState)
    {
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