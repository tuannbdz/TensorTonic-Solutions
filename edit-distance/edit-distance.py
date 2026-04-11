def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range (m + 1)]
    for i in range(m):
        dp[i + 1][0] = i + 1
    for i in range(n):
        dp[0][i + 1] = i + 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] != s2[j - 1]:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
            else:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]