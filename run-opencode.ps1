# load .env into the current process then launch opencode
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$OcArgs
)

$envFile = Join-Path $PSScriptRoot '.env'
if (Test-Path -LiteralPath $envFile) {
    Get-Content -LiteralPath $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#')) {
            if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
                $name = $matches[1]
                $value = $matches[2].Trim()
                if ($value.Length -ge 2) {
                    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                        ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                        $value = $value.Substring(1, $value.Length - 2)
                    }
                }
                if ($value.Length -gt 0) {
                    [Environment]::SetEnvironmentVariable($name, $value, 'Process')
                }
            }
        }
    }
}
else {
    Write-Warning "no .env found in $PSScriptRoot"
}

& opencode @OcArgs
exit $LASTEXITCODE