using CoreSumo;
using UnityEngine;

public static class BattleAPI
{
    public static float AngleToEnemy(this InputProvider i)
    {
        return Vector2.SignedAngle(Me(i).transform.up, (Enemy(i).transform.position - Me(i).transform.position).normalized);
    }
    public static Vector3 DistanceToEnemy(this InputProvider i)
    {
        return Enemy(i).transform.position - Me(i).transform.position;
    }

    public static Vector3 EnemyPosition(this InputProvider i)
    {
        return Enemy(i).transform.position;
    }

    public static SumoController Me(this InputProvider i)
    {
        if (i.PlayerSide == PlayerSide.Left)
        {
            return BattleManager.Instance.Battle.LeftPlayer;
        }
        else
        {
            return BattleManager.Instance.Battle.RightPlayer;
        }
    }

    public static SumoController Enemy(this InputProvider i)
    {
        if (i.PlayerSide == PlayerSide.Left)
        {
            return BattleManager.Instance.Battle.RightPlayer;
        }
        else
        {
            return BattleManager.Instance.Battle.LeftPlayer;
        }
    }


}