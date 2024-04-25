for ($i = 1; $i -le 5; $i++) {
    Write-Host "Iteration: $i"
    Invoke-Expression "python main.py"
}