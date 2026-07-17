using System;

namespace SumoLeaderboard
{
    /// <summary>
    /// Pure Elo math. Kept free of Unity types so it is trivially unit-testable.
    /// </summary>
    public static class EloCalculator
    {
        public const int KFactor = 32;
        public const int InitialRating = 1000;
        public const int RatingFloor = 100;

        public const float Win = 1f;
        public const float Draw = 0.5f;
        public const float Loss = 0f;

        public static float ExpectedScore(int ratingA, int ratingB)
        {
            return 1f / (1f + (float)Math.Pow(10, (ratingB - ratingA) / 400.0));
        }

        /// <summary>
        /// Computes both sides' new ratings. <paramref name="scoreA"/> is 1 (A won),
        /// 0.5 (draw) or 0 (A lost); B implicitly receives 1 - scoreA.
        /// </summary>
        public static (int newA, int newB) UpdatePair(int ratingA, int ratingB, float scoreA)
        {
            if (scoreA < 0f || scoreA > 1f)
                throw new ArgumentOutOfRangeException(nameof(scoreA));

            float expectedA = ExpectedScore(ratingA, ratingB);
            float expectedB = 1f - expectedA;
            float scoreB = 1f - scoreA;

            int newA = (int)Math.Round(ratingA + KFactor * (scoreA - expectedA));
            int newB = (int)Math.Round(ratingB + KFactor * (scoreB - expectedB));

            return (Math.Max(RatingFloor, newA), Math.Max(RatingFloor, newB));
        }
    }
}
