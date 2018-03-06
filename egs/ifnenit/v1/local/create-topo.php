<?php
/**
 * Create topology with 1 state silence phonemes and glyph dependent
 * model lengths for non-silence phonemes
 *
 * php create-topo.php <nstates-file> <non-silence-phonemes> <silence-phonemes>
 */

if (count($argv) != 4) {
  echo "Usage: php create-topo.php <nstates-file> <non-silence-phonemes> <silence-phonemes>\n";
  exit;
}
define('DEFAULT_NSTATES', 3);
define('MAX_NSTATES', 8);
$selfLoopProb = 0.75;
$changeProb = 0.25;

foreach (file($argv[2]) as $line) {
  $nStates[trim($line)] = DEFAULT_NSTATES;
}

foreach (file($argv[1]) as $line) {
  list($phoneme, $n) = explode(' ', trim($line), 2);
  $nStates[trim($phoneme)] = trim($n);
}

foreach (file($argv[3]) as $line) {
  $nStates[trim($line)] = 1;
}

$reverseLookup = array();
foreach ($nStates as $phoneme => $n) {
  if ($n > MAX_NSTATES) {
    $n = MAX_NSTATES;
  }
  if (!isset($reverseLookup[$n])) {
    $reverseLookup[$n] = array();
  }
  $reverseLookup[$n][] = $phoneme;
}

ksort($reverseLookup);

echo "<Topology>\n";
foreach ($reverseLookup as $n => $phonemes) {
  sort($phonemes);
  echo "<TopologyEntry>\n";
  echo "<ForPhones>\n";
  echo implode(' ', $phonemes) . "\n";
  echo "</ForPhones>\n";
  for ($curState = 0; $curState < $n; $curState++) {
    echo "<State> $curState <PdfClass> $curState <Transition> $curState $selfLoopProb <Transition> " 
      . ($curState+1) . " $changeProb </State>\n"; 
  }
  echo "<State> $curState </State>\n";
  echo "</TopologyEntry>\n";
}
echo "</Topology>\n";
