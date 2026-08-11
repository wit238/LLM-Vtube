# ===================================================
# Test OpenCode Go API Connection & Models
# ===================================================

# 1. Resolve API Key (ENV -> auth.json -> Default Key)
$key = $env:OPENCODE_API_KEY

if (-not $key) {
    $authPath = "$env:USERPROFILE\.local\share\opencode\auth.json"
    if (Test-Path -LiteralPath $authPath) {
        try {
            $authJson = Get-Content -LiteralPath $authPath -Raw | ConvertFrom-Json
            if ($authJson.'opencode-go' -and $authJson.'opencode-go'.key) {
                $key = $authJson.'opencode-go'.key
            }
        } catch {}
    }
}

if (-not $key) {
    $key = "sk-DFrcqkVPP6sirRHKyKjQteUBYmMzxaT6PgAGQHnSDSAEp99kyVB9k0dgUvLlxjX0"
}

$url = "https://opencode.ai/zen/v1/chat/completions"
$models = @("deepseek-v4-flash-free", "deepseek-v4-flash")

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " OpenCode Go API Tester" -ForegroundColor Cyan
Write-Host " URL: $url" -ForegroundColor DarkGray
Write-Host " Key: $key" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Cyan

foreach ($model in $models) {
    Write-Host "`n---------------------------------------------------" -ForegroundColor DarkGray
    Write-Host " Testing Model: $model" -ForegroundColor Magenta
    Write-Host "---------------------------------------------------" -ForegroundColor DarkGray
    
    $headers = @{
        "Authorization" = "Bearer $key"
        "Content-Type"  = "application/json"
    }

    $body = @{
        model = $model
        messages = @(
            @{ role = "user"; content = "Hello! Test connection. Reply in 5 words." }
        )
    } | ConvertTo-Json -Depth 5

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        $res = Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $body
        $stopwatch.Stop()

        Write-Host " STATUS   : SUCCESS" -ForegroundColor Green
        Write-Host " TIME     : $($stopwatch.ElapsedMilliseconds) ms" -ForegroundColor Gray
        Write-Host " MODEL    : $($res.model)" -ForegroundColor DarkGreen
        Write-Host " RESPONSE : $($res.choices[0].message.content)" -ForegroundColor White

        if ($res.usage) {
            Write-Host " TOKENS   : Prompt=$($res.usage.prompt_tokens), Completion=$($res.usage.completion_tokens), Total=$($res.usage.total_tokens)" -ForegroundColor DarkGray
        }
    } catch {
        $stopwatch.Stop()
        Write-Host " STATUS   : FAILED (HTTP $($_.Exception.Response.StatusCode.value__))" -ForegroundColor Red
        Write-Host " TIME     : $($stopwatch.ElapsedMilliseconds) ms" -ForegroundColor Gray
        
        if ($_.Exception.Response) {
            $stream = $_.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($stream)
            $respBody = $reader.ReadToEnd()
            Write-Host " ERROR    : $respBody" -ForegroundColor Yellow
        } else {
            Write-Host " ERROR    : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

Write-Host "`n===================================================" -ForegroundColor Cyan
