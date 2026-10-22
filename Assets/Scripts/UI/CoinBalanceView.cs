using SumoServices;
using TMPro;
using UnityEngine;

// Shows the signed-in player's coin balance and keeps itself in sync by subscribing to
// IPlayerDataService.CoinsChanged. Nothing has to push updates after a purchase — the
// service raises the event and every live CoinBalanceView refreshes on its own.
//
// Attach to each place the balance is shown (top-bar PlayerMoney, Market PlayerMoney);
// each instance is independent. Assign the TMP_Text that displays the number.
public class CoinBalanceView : MonoBehaviour
{
    [SerializeField] private TMP_Text balanceText;

    // Cache the exact service we subscribed to so OnDisable always detaches from the same
    // instance, even if GameServices were reset (test hook) between enable and disable.
    private IPlayerDataService subscribed;

    void OnEnable()
    {
        subscribed = GameServices.PlayerData;
        if (subscribed != null)
        {
            subscribed.CoinsChanged += Refresh;
            Refresh();
        }
    }

    void OnDisable()
    {
        if (subscribed != null)
        {
            subscribed.CoinsChanged -= Refresh;
            subscribed = null;
        }
    }

    private void Refresh()
    {
        if (balanceText == null) return;
        balanceText.text = (subscribed?.Current?.Coins ?? 0).ToString();
    }
}
