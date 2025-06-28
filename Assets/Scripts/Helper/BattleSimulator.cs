using System.Collections;
using UnityEngine;

public class BattleSimulator : MonoBehaviour
{
    public int TotalSimulations = 5;
    public float TimeScale = 5f;

    // Should be [false] if we ewant to run in Headless-mode
    public bool SimulationOnStart = false;



    void Start()
    {
        if (SimulationOnStart)
            StartSimulation();
    }

    public void StartSimulation()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = true;
#endif

        Time.timeScale = TimeScale;
        Application.runInBackground = true;
        // Time.fixedDeltaTime = 0.02f / TimeScale;
        StartCoroutine(RunSimulations());
    }

    private IEnumerator RunSimulations()
    {
        //Delay for preparing
        yield return new WaitForSeconds(0.5f);

        for (int i = 0; i < TotalSimulations; i++)
        {
            yield return new WaitForSeconds(1);
            BattleManager.Instance.Battle_Start();

            // Wait until match is over
            while (BattleManager.Instance.CurrentState != BattleState.PostBattle_ShowResult)
            {
                yield return null; // wait frame
            }
            
            yield return new WaitForSeconds(1);
            yield return new WaitForEndOfFrame(); // Delay if needed
        }

        Debug.Log("Simulation complete.");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }
}
