using System;
using NUnit.Framework;

using Unity.VisualScripting;
using UnityEngine;


namespace Test
{

    public class InputHandlingTest
    {

        [Test]
        public void RobotDashTest()
        {

            var robot = new GameObject()
            .AddComponent<Rigidbody2D>()
            .AddComponent<Robot>();

            robot.Initalize();

            robot.GetComponent<Rigidbody2D>().linearVelocity = Vector2.zero;

            Assert.Throws<Exception>(()=>robot.Dash());
        }



        [Test]
        public void RobotCalculateEnemyDist()
        {

            var robot = new GameObject()
            .AddComponent<Robot>();

            Assert.Throws<Exception>(() => robot.CalcuateEnemyDistance());
        }

    }


}