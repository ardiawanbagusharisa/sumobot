using UnityEngine;

public class DebounceButtonState
{
    private float debounceTime;
    private float lastCallTime;
    public bool IsActive;
    private float startTime;
    private string Name;
    private GameObject go;

    public DebounceButtonState(GameObject go, float debounceTime)
    {
        this.debounceTime = debounceTime;
        this.go = go;
    }

    public void Call(string name)
    {
        Name = name;
        if (!IsActive)
        {
            IsActive = true;
            startTime = BattleManager.Instance.ElapsedTime;
        }

        lastCallTime = BattleManager.Instance.ElapsedTime;
    }

    public void Update()
    {
        if (IsActive && BattleManager.Instance.ElapsedTime - lastCallTime >= debounceTime)
        {
            IsActive = false;
        }
    }
}
