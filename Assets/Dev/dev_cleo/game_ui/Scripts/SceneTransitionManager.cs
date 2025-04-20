using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneTransitionManager : MonoBehaviour
{
    public Animator animator;

    public void StartSceneTransition(string sceneName)
    {
        StartCoroutine(PlayTransition(sceneName));
    }

    IEnumerator PlayTransition(string sceneName)
    {
        animator.SetTrigger("End"); // Mulai dari END
        yield return new WaitForSeconds(1f); // durasi Trans_End
        animator.SetTrigger("Start"); // Lanjut ke START
        yield return new WaitForSeconds(1f); // durasi Trans_Start
        SceneManager.LoadScene(sceneName);
    }
}
