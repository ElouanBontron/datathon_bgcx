# Met a jour le clone complet ou la copie scientifique dans upstream/datathon_bgcx.
# Aucune suppression ni reinitialisation Git ; arret si des changements locaux existent.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$repositoryUrl = 'https://github.com/ElouanBontron/datathon_bgcx.git'
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$upstreamPath = Join-Path -Path $workspacePath -ChildPath 'upstream'
$isFullClone = Test-Path -LiteralPath (Join-Path -Path $workspacePath -ChildPath '.git')
$repositoryPath = if ($isFullClone) {
    $workspacePath
} else {
    Join-Path -Path $upstreamPath -ChildPath 'datathon_bgcx'
}

try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'Git est absent. Installez Git pour Windows, puis relancez ce script.'
    }

    if (-not (Test-Path -LiteralPath $repositoryPath)) {
        if (-not (Test-Path -LiteralPath $upstreamPath)) {
            New-Item -ItemType Directory -Path $upstreamPath | Out-Null
        }
        Write-Host 'Téléchargement du dépôt GitHub du hackathon…'
        & git clone --depth 1 -- $repositoryUrl $repositoryPath
        if ($LASTEXITCODE -ne 0) {
            throw 'Le téléchargement a échoué. Vérifiez la connexion et le message de Git ci-dessus.'
        }
    }
    else {
        if (-not (Test-Path -LiteralPath $repositoryPath -PathType Container)) {
            throw "Le chemin du dépôt existe mais n'est pas un dossier : $repositoryPath"
        }
        $gitRoot = & git -C $repositoryPath rev-parse --show-toplevel
        if ($LASTEXITCODE -ne 0) {
            throw "Ce dossier n'est pas une copie Git valide : $repositoryPath"
        }
        $resolvedGitRoot = [System.IO.Path]::GetFullPath([string]$gitRoot)
        $expectedGitRoot = [System.IO.Path]::GetFullPath($repositoryPath)
        if (-not $resolvedGitRoot.Equals($expectedGitRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Le dossier appartient à un autre dépôt Git. Aucune mise à jour effectuée.'
        }
        $originUrl = & git -C $repositoryPath remote get-url origin
        if ($LASTEXITCODE -ne 0) {
            throw "Le dépôt ne possède pas d'origine Git lisible. Aucune mise à jour effectuée."
        }
        $expectedOrigin = '^(https://github\.com/ElouanBontron/datathon_bgcx(?:\.git)?/?|git@github\.com:ElouanBontron/datathon_bgcx(?:\.git)?)$'
        if ([string]$originUrl -notmatch $expectedOrigin) {
            throw "L'origine ne correspond pas au dépôt du hackathon : $originUrl"
        }
        $localChanges = @(& git -C $repositoryPath status --porcelain=v1 --untracked-files=normal)
        if ($LASTEXITCODE -ne 0) {
            throw "Impossible de vérifier les modifications locales. Aucune mise à jour effectuée."
        }
        if ($localChanges.Count -gt 0) {
            Write-Host ($localChanges -join [Environment]::NewLine)
            throw 'Le d?p?t contient des modifications locales. Conservez ou validez ces changements avant de relancer ; aucun fichier ne sera écrasé.'
        }
        Write-Host 'Mise à jour autorisée uniquement par avance rapide (fast-forward)…'
        & git -C $repositoryPath pull --ff-only
        if ($LASTEXITCODE -ne 0) {
            throw 'Git a refusé la mise à jour. Vérifiez la connexion ou une divergence de branches ; aucune fusion ni réinitialisation automatique effectuée.'
        }
    }

    $commit = & git -C $repositoryPath rev-parse --short HEAD
    if ($LASTEXITCODE -ne 0) {
        throw 'Le téléchargement est terminé, mais le commit local ne peut pas être lu.'
    }
    Write-Host "Données disponibles au commit $commit."
    if ($isFullClone) {
        Write-Host "Depot complet a jour. Redemarrez FlowPilot pour recharger le code et les donnees."
    } else {
        Write-Host "Copie scientifique a jour ; interface locale conservee. Redemarrez FlowPilot."
    }
}
catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
