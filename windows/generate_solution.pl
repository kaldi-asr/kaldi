#!/usr/bin/perl

#
# Copyright:2012-2015 Jan Silovsky
#           2015 Johns Hopkins University (Author: Jan "Yenda" Trmal <jtrmal@gmail.com>)

# Licence: Apache 2.0
# Copyright Jan Silovsky.
# Modifications and small bugfixes by Jan "Yenda" Trmal

# This program generates the Visual Studio solution file.
# Note, this program has been tested with both cygwin and
# Windows (ActiveState perl) versions of Perl

use strict;
use File::Find;
use File::Path;
use File::Copy;
use FindBin qw($Bin);
use lib "$Bin";
use Data::Dumper;
use Getopt::Long;

my $vsver="vs2017";

my %ENABLED = (CUDA => 0,
               OPENBLAS => 0,
               MKL => 1 );

GetOptions ("vsver=s" => \$vsver,
            "enable-cuda" => \$ENABLED{CUDA},
			"enable-openblas" => sub {$ENABLED{OPENBLAS}=1; $ENABLED{MKL}=0;},
			"enable-mkl" => sub {$ENABLED{OPENBLAS}=0; $ENABLED{MKL}=1;},
			);

my %TOOLS=( default=> "14.1",
            vs2015 => "14.0",
            vs2017 => "14.1"
            );

my %FORMAT=( default=> "14.10",
             vs2015 =>  "14.00",
             vs2017 =>  "14.10"
             );

my %TOOLSET=( default=> "v141",
              vs2015 => "v140",
              vs2017 => "v141"
              );


unless ((defined $TOOLS{$vsver}) && (defined $FORMAT{$vsver}) && (defined $TOOLSET{$vsver})) {
	die "Unknown vsver value: $vsver";
}

my $features_suffix;
my @features_enabled;

if ($ENABLED{OPENBLAS}) {
  push @features_enabled, "OPENBLAS";
}
if ($ENABLED{MKL}) {
  push @features_enabled, "MKL";
}
if ($ENABLED{CUDA}) {
  push @features_enabled, "CUDA";
}

$features_suffix = join("_", @features_enabled);

my $root = "$Bin/..";
my $solutionDir = "$root/kaldiwin_${vsver}_${features_suffix}";
my $projDir = "$solutionDir/kaldiwin";
my $solutionFileName = "kaldiwin_${vsver}.sln";
my $srcDir = "$root/src";


# The following files are in the same dir (windows/) as the
# Perl script.
my @propsFiles = (
  "$Bin/variables.props",
  "$Bin/kaldiwin.props",
  "$Bin/kaldiwin_win32.props",
  "$Bin/openfstwin_debug.props",
  "$Bin/openfstwin_release.props",
  "$Bin/openfstwin_debug_win32.props",
  "$Bin/openfstwin_release_win32.props",
);

my %optionalProps = (
	CUDA => "$Bin/cuda_7.0.props"
	);

# see http://www.mztools.com/Articles/2008/MZ2008017.aspx for list of GUIDs for VS solutions
my $globalGUID = "{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}";  # Windows (Visual C++)

my $projguidListFileName = "$Bin/kaldiwin_projguids.txt";
my $guidgen = "$Bin/NewGuidCmd.exe";  # it is C# application

my $osPathConversion = \&winPath;
if ($^O !~ /MSWin32/i) {
  $osPathConversion = \&unixPath;
}

# function
sub winPath {
  my $path = shift;
  $path =~ s/\//\\/g;
  return $path;
}

# function
sub unixPath {
  my $path = shift;
  $path =~ s/\\/\//g;
  return $path;
}

# function
sub removeStartSlash {
  my $path = shift;
  $path =~ s/^[\\\/]//;
  return $path;
}

sub makeRelPath {
  my $path = winPath(shift);
  my $actual_path = winPath(shift);
  my $relpath = '';

  $path =~ s/\\cygdrive\\(.)\\/$1:\\/;
  $actual_path =~ s/\\cygdrive\\(.)\\/$1:\\/;

  #$path =~ s/[\\\/]+[^\\\/]*$//;
  $actual_path =~ s/[\\\/]+[^\\\/]*$//;

  my @path = split /\\+/, $path;
  my @actual_path = split /\\+/, $actual_path;

  my $i = 0;
  for ($i; $i < @actual_path; $i++) {
    # the zero-length condition may seem ackward, but it solves encountered troubles when two empty string were evaluated as different
    # the case-insensitive comparison is used because of Windows filesystem case-insensitivity
    if (($path[$i] !~ /$actual_path[$i]/i) && (length($path[$i]) > 0) && (length($actual_path[$i]) > 0)) {
      last;
    }
  }
  my $j = $i;
  for ($i; $i < @actual_path; $i++) {
    $relpath .= "..\\";
  }
  for ($j; $j < (@path - 1); $j++) {
    $relpath .= $path[$j] . "\\";
  }
  $relpath .= $path[$j];

  return $relpath;
}

# function
sub checkCRLF {
  my $filename = shift;

  if ($^O =~ /MSWin32/i) {
    print "INFO: function checkCRLF supported only for non-Win environment\n";
    return;
  }

  open(FILE, '<', $filename);
  my @data = <FILE>;
  close(FILE);

  open(FILE, '>', $filename);
  foreach my $line (@data) {
    $line =~ s/(\n|\r\n)$//;
    print FILE $line . "\r\n";
  }
  close(FILE);
}

# function
sub loadHashTxtFile {
  my $filename = shift;
  my $hash = shift;

  open(FILE, '<', &{$osPathConversion}($filename));
  while (my $line = <FILE>) {
    $line =~ s/(\n|\r\n)$//;
    if (my ($key, $value) = $line =~ /^(\S+)\s+(\S+)$/) {
      $hash->{$key} = $value;
    } else {
      print STDERR "ERROR: unable to parse line $line in file $filename\n";
    }
  }
  close(FILE);
}

# function
sub saveHashTxtFile {
  my $filename = shift;
  my $hash = shift;

  open(FILE, '>', &{$osPathConversion}($filename));
  foreach my $key (sort { $a cmp $b } keys %$hash) {
    print FILE $key . "\t" . $hash->{$key} . "\n";
  }
  close(FILE);
}

# function
sub parseMakefile {
  my $makefile = shift;
  my $list = shift;
  my $deps = shift;
  my $libs = shift;

  my $file = $makefile;
  my $path = $file;
  $path =~ s/Makefile$//i;

  open(FILE, '<', &{$osPathConversion}($file));
  my @lines = <FILE>;
  close(FILE);

  my $lines = join '', @lines;
  $lines =~ s/#[^\n]+//g;
  $lines =~ s/\\\s*\n//g;
  if ($ENABLED{CUDA}) {
    $lines =~ s/\n\s*ifeq\s+\(\$\(CUDA\),\strue\)\s*\n\s*OBJFILES\s*\+=\s*([^\n]+)\n\s*endif/\nOBJFILES = \1/gmi
  }
  @lines = split /\n/, $lines;
  #$lines =~ s/\\//g;
  #$lines =~ s/\n/\\/g;
  #print $lines . "\n";
  #@lines = split /\\/, $lines;

  my $aobjects = {};
  my $alibs = {};
  foreach my $line (@lines) {
    $line =~ s/(\n|\r\n)$//;

    if (my ($type, $items) = $line =~ /^\s*(TESTFILES|LIBNAME|BINFILES)\s+=(.+?)$/) {
      #my @items = split /\s+/, $items;
      my @items = $items =~ /(\S+)/g;
      foreach my $item (@items) {
        # $item =~ s/\.a$//i;

        $list->{$type}->{$item} = 1;
        $list->{ALL}->{$item}->{'type'} = $type;
        $list->{ALL}->{$item}->{'path'} = $path;

        if ($type =~ /LIBNAME/) {
          $alibs->{$item} = 1;
        }
      }
    }

    if (my ($items) = $line =~ /^ADDLIBS[^=]*?=(.+?)$/) {
	  my @items = $items =~ /(\S+)/g;

	  foreach my $alib (keys %$alibs) {
		$deps->{$path}->{$alib} = 1;
	  }
      foreach my $item (@items) {
        $item =~ s/^.*[\\\/]//;
        $item =~ s/\.[^\.]*$//;

        $deps->{$path}->{$item} = 1;

      }
    }

    if (my ($type, $items) = $line =~ /^\s*(OBJFILES)\s+=(.+?)$/) {
      my @items = $items =~ /(\S+)/g;
      foreach my $item (@items) {
        $aobjects->{$item} = 1;
      }
    }
  }

  if (scalar keys %{$aobjects} > 0) {
    if (scalar keys %{$alibs} != 1) {
      print STDERR "ERROR: less or more than one libfile, cannot assign aobjects\n";
    } else {
      my $alib = join('', keys %{$alibs}); # $alibs obsahuje nazev pouze jedne knihovny
      foreach my $obj (keys %{$aobjects}) {
        $obj =~ s/\.o$//i;
        $list->{ALL}->{$alib}->{'objs'}->{$obj} = 1;
      }
    }
  }
}

# function
sub makefileFilter {
  if ($_ =~ /Makefile/) { return 1; }
}

# function
sub getProjFileDir {
  my $projname = shift;
  my $projFileName = $projDir . "/$projname";
  return $projFileName;
}

# function
sub writeSolutionFile {
  my $filename = shift;
  my $projlist = shift;
  my $projguids = shift;

  open(SLN, '>', &{$osPathConversion}($filename));
  print SLN
"Microsoft Visual Studio Solution File, Format Version $FORMAT{$vsver}
# Visual Studio 2013
";
  foreach my $projname (sort { $a cmp $b } keys %{$projlist->{ALL}}) {
    if (!exists $projguids->{$projname}) {
      my $cmd = "$guidgen";
      my $guid = `$cmd`;
      $guid =~ s/(\n|\r\n)$//;
      $projguids->{$projname} = uc('{' . $guid . '}');
    }
    my $guid = $projguids->{$projname};
    $projlist->{ALL}->{$projname}->{'guid'} = $guid;
    my $projFileName = winPath(makeRelPath(getProjFileDir($projname), $filename) . "/$projname.vcxproj");
    print SLN
"Project(\"$globalGUID\") = \"$projname\", \"$projFileName\", \"$guid\"
EndProject
";
  }
  print SLN
"Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|x64 = Debug|x64
		Debug|Win32 = Debug|Win32
		Release|x64 = Release|x64
		Release|Win32 = Release|Win32
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
";
  foreach my $projname (sort { $a cmp $b } keys %{$projlist->{ALL}}) {
    my $guid = $projlist->{ALL}->{$projname}->{'guid'};
    print SLN
"		$guid.Debug|Win32.ActiveCfg = Debug|Win32
		$guid.Debug|Win32.Build.0 = Debug|Win32
		$guid.Debug|x64.ActiveCfg = Debug|x64
		$guid.Debug|x64.Build.0 = Debug|x64
		$guid.Release|Win32.ActiveCfg = Release|Win32
		$guid.Release|Win32.Build.0 = Release|Win32
		$guid.Release|x64.ActiveCfg = Release|x64
		$guid.Release|x64.Build.0 = Release|x64
";
  }
  print SLN
"	EndGlobalSection
	GlobalSection(SolutionProperties) = preSolution
		HideSolutionNode = FALSE
	EndGlobalSection
EndGlobal
";
  close(SLN);
}

# function
sub writeProjectFiles {
  my $projname = shift;
  my $projlist = shift;
  my $projdeps = shift;
  my $projlibs = shift;
  my $projguids = shift;

  my $projFileName = winPath(getProjFileDir($projname) . "/$projname.vcxproj");

  my $guid = $projguids->{$projname};
  my $rootnamespace = $projname;
  $rootnamespace =~ s/\W+//g;
  my $srcfiles = {};
  my $conftype = "";

  # set projtype-specific params and add .cc files
  if ($projlist->{ALL}->{$projname}->{'type'} =~ /LIBNAME/) {
    $conftype = "StaticLibrary";

    foreach my $obj (keys %{$projlist->{ALL}->{$projname}->{'objs'}}) {
      my $cfile = winPath($projlist->{ALL}->{$projname}->{'path'} . $obj . '.cc');
	  if (!-e &{$osPathConversion}($cfile)) {
	    if ($ENABLED{CUDA}) {
          my $cufile = winPath($projlist->{ALL}->{$projname}->{'path'} . $obj . '.cu');
          $srcfiles->{'cu'}->{$cufile} = 1;
          if (!-e &{$osPathConversion}($cufile)) {
		    print "ERROR: file $cfile nor $cufile not found - project $projname\n";
		  }
        } else {
          if (!-e &{$osPathConversion}($cfile)) {
		    print "ERROR?: file $cfile not found - project $projname\n";
		  }
        }
      } else {
        $srcfiles->{'cc'}->{$cfile} = 1;
	  }
    }
  } else {
    $conftype = "Application";

    my $cfile = winPath($projlist->{ALL}->{$projname}->{'path'} . $projname . '.cc');
    $srcfiles->{'cc'}->{$cfile} = 1;
  }

  # add .h files + check that .cc files exist
  foreach my $cfile (keys %{$srcfiles->{'cc'}}) {
    if (!-e &{$osPathConversion}($cfile)) {
      print "ERROR: file $cfile not found - project $projname\n";
    }
    my $hfile = $cfile;
    $hfile =~ s/\.[^\.]+$/.h/;
    if (-e &{$osPathConversion}($hfile)) {
      $srcfiles->{'h'}->{$hfile} = 1;
    }
    my $hinlfile = $cfile;
    $hinlfile =~ s/\.[^\.]+$/-inl.h/;
    if (-e &{$osPathConversion}($hinlfile)) {
      $srcfiles->{'h'}->{$hinlfile} = 1;
    }
  }

  open(PROJ, '>', &{$osPathConversion}($projFileName));
  print PROJ
"<?xml version=\"1.0\" encoding=\"utf-8\"?>
<Project DefaultTargets=\"Build\" ToolsVersion=\"$TOOLS{$vsver}\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">
  <ItemGroup Label=\"ProjectConfigurations\">
    <ProjectConfiguration Include=\"Debug|Win32\">
      <Configuration>Debug</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include=\"Debug|x64\">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include=\"Release|Win32\">
      <Configuration>Release</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include=\"Release|x64\">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label=\"Globals\">
    <ProjectGuid>" . $guid . "</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>" . $rootnamespace . "</RootNamespace>
  </PropertyGroup>
";

  if ($projlist->{ALL}->{$projname}->{'type'} =~ /LIBNAME/) { # Microsoft.Cpp.Default.props - Library
    print PROJ
"  <Import Project=\"\$(VCTargetsPath)\\Microsoft.Cpp.Default.props\" />
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\" Label=\"Configuration\">
    <ConfigurationType>" . $conftype . "</ConfigurationType>
    <CharacterSet>Unicode</CharacterSet>
    <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|x64'\" Label=\"Configuration\">
    <ConfigurationType>" . $conftype . "</ConfigurationType>
    <CharacterSet>Unicode</CharacterSet>
    <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\" Label=\"Configuration\">
    <ConfigurationType>" . $conftype . "</ConfigurationType>
    <CharacterSet>Unicode</CharacterSet>
    <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|x64'\" Label=\"Configuration\">
     <ConfigurationType>" . $conftype . "</ConfigurationType>
     <CharacterSet>Unicode</CharacterSet>
     <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
     <WholeProgramOptimization>true</WholeProgramOptimization>
  </PropertyGroup>
";
  } else {  # Microsoft.Cpp.Default.props - Binfile
    print PROJ
"  <Import Project=\"\$(VCTargetsPath)\\Microsoft.Cpp.Default.props\" />
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\" Label=\"Configuration\">
    <ConfigurationType>" . $conftype . "</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <CharacterSet>Unicode</CharacterSet>
    <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\" Label=\"Configuration\">
    <ConfigurationType>" . $conftype . "</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
    <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|x64'\" Label=\"Configuration\">
    <ConfigurationType>" . $conftype . "</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <CharacterSet>Unicode</CharacterSet>
    <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|x64'\" Label=\"Configuration\">
    <ConfigurationType>" . $conftype . "</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
    <PlatformToolset>$TOOLSET{$vsver}</PlatformToolset>
  </PropertyGroup>
";
  }

  print PROJ
"  <Import Project=\"\$(VCTargetsPath)\\Microsoft.Cpp.props\" />
  <ImportGroup Label=\"ExtensionSettings\">
";
  if ($ENABLED{CUDA}) {
  print PROJ
'    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 7.0.props" />
'
  }
  print PROJ
"  </ImportGroup>
  <ImportGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\"  Label=\"PropertySheets\">
    <Import Project=\"..\\variables.props\" />
    <Import Project=\"\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props\" Condition=\"exists('\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />
";
  if ($ENABLED{CUDA}) {
  print PROJ
"    <Import Project=\"..\\cuda_7.0.props\" />
"
  }
  print PROJ
"    <Import Project=\"..\\kaldiwin_win32.props\" />
    <Import Project=\"..\\openfstwin_debug_win32.props\" />
  </ImportGroup>
  <ImportGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|x64'\" Label=\"PropertySheets\">
    <Import Project=\"..\\variables.props\" />
    <Import Project=\"\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props\" Condition=\"exists('\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />
";
  if ($ENABLED{CUDA}) {
  print PROJ
"    <Import Project=\"..\\cuda_7.0.props\" />
";
  }
  print PROJ
"    <Import Project=\"..\\kaldiwin.props\" />
    <Import Project=\"..\\openfstwin_debug.props\" />
  </ImportGroup>
  <ImportGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\" Label=\"PropertySheets\">
    <Import Project=\"..\\variables.props\" />
    <Import Project=\"\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props\" Condition=\"exists('\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />
";
  if ($ENABLED{CUDA}) {
  print PROJ
"    <Import Project=\"..\\cuda_7.0.props\" />
";
  }
  print PROJ
"    <Import Project=\"..\\kaldiwin_win32.props\" />
    <Import Project=\"..\\openfstwin_release_win32.props\" />
  </ImportGroup>
  <ImportGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|x64'\" Label=\"PropertySheets\">
    <Import Project=\"..\\variables.props\" />
    <Import Project=\"\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props\" Condition=\"exists('\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />
";
  if ($ENABLED{CUDA}) {
  print PROJ
"    <Import Project=\"..\\cuda_7.0.props\" />
";
  }
  print PROJ
"    <Import Project=\"..\\kaldiwin.props\" />
    <Import Project=\"..\\openfstwin_release.props\" />
  </ImportGroup>
";

  if ($projlist->{ALL}->{$projname}->{'type'} =~ /LIBNAME/) { # UserMacros - Library
    print PROJ
"  <PropertyGroup Label=\"UserMacros\" />
  <PropertyGroup>
    <_ProjectFileVersion>10.0.30319.1</_ProjectFileVersion>
    <OutDir Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\">\$(SolutionDir)\$(Configuration)\\</OutDir>
    <IntDir Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\">\$(Configuration)\\</IntDir>
    <OutDir Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\">\$(SolutionDir)\$(Configuration)\\</OutDir>
    <IntDir Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\">\$(Configuration)\\</IntDir>
  </PropertyGroup>
";
  } else {  # UserMacros - Binfile
    print PROJ
"  <PropertyGroup Label=\"UserMacros\" />
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\">
    <LinkIncremental>true</LinkIncremental>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|x64'\">
    <LinkIncremental>true</LinkIncremental>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\">
    <LinkIncremental>false</LinkIncremental>
  </PropertyGroup>
  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|x64'\">
    <LinkIncremental>false</LinkIncremental>
  </PropertyGroup>
";
  }

  if ($projlist->{ALL}->{$projname}->{'type'} =~ /LIBNAME/) { # ItemDefinitionGroup Conditions - Library
    print PROJ
"  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\">
    <ClCompile>
      <Optimization>Disabled</Optimization>
      <PreprocessorDefinitions>WIN32;_DEBUG;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <MinimalRebuild>true</MinimalRebuild>
      <BasicRuntimeChecks>EnableFastChecks</BasicRuntimeChecks>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>EditAndContinue</DebugInformationFormat>
    </ClCompile>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|x64'\">
    <ClCompile>
      <Optimization>Disabled</Optimization>
      <PreprocessorDefinitions>WIN32;_DEBUG;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <MinimalRebuild>true</MinimalRebuild>
      <BasicRuntimeChecks>EnableFastChecks</BasicRuntimeChecks>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
    </ClCompile>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\">
    <ClCompile>
      <Optimization>MaxSpeed</Optimization>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;NDEBUG;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
    </ClCompile>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|x64'\">
    <ClCompile>
      <Optimization>MaxSpeed</Optimization>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;NDEBUG;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>
    </ClCompile>
  </ItemDefinitionGroup>
";
  } else {  # ItemDefinitionGroup Conditions - Binfile
    print PROJ
"  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|Win32'\">
    <ClCompile>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>Disabled</Optimization>
      <PreprocessorDefinitions>WIN32;_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
   <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Debug|x64'\">
     <ClCompile>
       <PrecompiledHeader>
       </PrecompiledHeader>
       <WarningLevel>Level3</WarningLevel>
       <Optimization>Disabled</Optimization>
       <PreprocessorDefinitions>WIN32;_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
     </ClCompile>
     <Link>
       <SubSystem>Console</SubSystem>
       <GenerateDebugInformation>true</GenerateDebugInformation>
     </Link>
   </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|Win32'\">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='Release|x64'\">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
    </Link>
  </ItemDefinitionGroup>
";
  }

  # .c files
  if (scalar keys %{$srcfiles->{'cc'}} > 0) {
    print PROJ
"  <ItemGroup>
";
    foreach my $cfile (sort { $a cmp $b } keys %{$srcfiles->{'cc'}}) {
      print PROJ
"    <ClCompile Include=\"" . makeRelPath($cfile, $projFileName) . "\" />
";
    }
    print PROJ
"  </ItemGroup>
";
  }
  # .cu files
  if (scalar keys %{$srcfiles->{'cu'}} > 0) {
    print PROJ
"  <ItemGroup>
";
    foreach my $cfile (sort { $a cmp $b } keys %{$srcfiles->{'cu'}}) {
      print PROJ
"    <CudaCompile Include=\"" . makeRelPath($cfile, $projFileName) . "\" />
";
    }
    print PROJ
"  </ItemGroup>
";
  }

  # .h files
  if (scalar keys %{$srcfiles->{'h'}} > 0) {
    print PROJ
"  <ItemGroup>
";
    foreach my $hfile (sort { $a cmp $b } keys %{$srcfiles->{'h'}}) {
      print PROJ
"    <ClInclude Include=\"" . makeRelPath($hfile, $projFileName) . "\" />
";
    }
    print PROJ
"  </ItemGroup>
";
  }

  # refs
  if (($projlist->{ALL}->{$projname}->{'type'} !~ /LIBNAME/) &&
      (scalar keys %{$projdeps->{$projlist->{ALL}->{$projname}->{'path'}}} > 0)) {
    print PROJ
"  <ItemGroup>
";
    foreach my $lib (sort { $a cmp $b } keys%{$projdeps->{$projlist->{ALL}->{$projname}->{'path'}}}) {
      my $refProjFileName = makeRelPath(winPath(getProjFileDir($lib) . "/$lib.vcxproj"), $projFileName);
      print PROJ
"    <ProjectReference Include=\"" . $refProjFileName . "\">
      <Project>" .   lc($projguids->{$lib}) . "</Project>
    </ProjectReference>
";
    }
    print PROJ
"  </ItemGroup>
";
  }

  # terminate
  print PROJ
"  <Import Project=\"\$(VCTargetsPath)\\Microsoft.Cpp.targets\" />
  <ImportGroup Label=\"ExtensionTargets\">
";
  if ($ENABLED{CUDA}) {
    print PROJ
'    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 7.0.targets" />
';
  }
  print PROJ
"  </ImportGroup>
</Project>
";
  close(PROJ);

  # create .user file
  #   my $filename_userfile = $projFileName . '.user';
    # open(USER, '>', &{$osPathConversion}($filename_userfile));
    # print USER
  # "<?xml version=\"1.0\" encoding=\"utf-8\"?>
  # <Project ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">
  # </Project>
  # ";
    # close(USER);

  # create .filters file
  my $filename_filtersfile = $projFileName . '.filters';
  open(FLTS, '>', &{$osPathConversion}($filename_filtersfile));
  print FLTS
"<?xml version=\"1.0\" encoding=\"utf-8\"?>
<Project ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">
  <ItemGroup>
    <Filter Include=\"Source Files\">
      <UniqueIdentifier>{4FC737F1-C7A5-4376-A066-2A32D752A2FF}</UniqueIdentifier>
      <Extensions>cpp;c;cc;cxx;def;odl;idl;hpj;bat;asm;asmx</Extensions>
    </Filter>
    <Filter Include=\"Header Files\">
      <UniqueIdentifier>{93995380-89BD-4b04-88EB-625FBE52EBFB}</UniqueIdentifier>
      <Extensions>h;hpp;hxx;hm;inl;inc;xsd</Extensions>
    </Filter>
    <Filter Include=\"Resource Files\">
      <UniqueIdentifier>{67DA6AB6-F800-4c08-8B7A-83BB121AAD01}</UniqueIdentifier>
      <Extensions>rc;ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe;resx;tiff;tif;png;wav;mfcribbon-ms</Extensions>
    </Filter>
    <Filter Include=\"Cuda Kernels\">
      <UniqueIdentifier>{6841a02e-469b-487d-a1ea-7d138415dd41}</UniqueIdentifier>
      <Extensions>cu</Extensions>
    </Filter>
  </ItemGroup>
";
  # .c files
  if (scalar keys %{$srcfiles->{'cc'}} > 0) {
    print FLTS
"  <ItemGroup>
";
    foreach my $cfile (sort { $a cmp $b } keys %{$srcfiles->{'cc'}}) {
      print FLTS
"    <ClCompile Include=\"" . makeRelPath($cfile, $projFileName) . "\">
      <Filter>Source Files</Filter>
    </ClCompile>
";
    }
    print FLTS
"  </ItemGroup>
";
  }
  # .h files
  if (scalar keys %{$srcfiles->{'h'}} > 0) {
    print FLTS
"  <ItemGroup>
";
    foreach my $hfile (sort { $a cmp $b } keys %{$srcfiles->{'h'}}) {
      print FLTS
"    <ClInclude Include=\"" . makeRelPath($hfile, $projFileName) . "\">
      <Filter>Header Files</Filter>
    </ClInclude>
";
    }
    print FLTS
"  </ItemGroup>
";
  }
  # .cu files
  if (scalar keys %{$srcfiles->{'cu'}} > 0) {
    print FLTS
"  <ItemGroup>
";
    foreach my $cufile (sort { $a cmp $b } keys %{$srcfiles->{'cu'}}) {
      print FLTS
"    <CudaCompile Include=\"" . makeRelPath($cufile, $projFileName) . "\">
      <Filter>Cuda Kernels</Filter>
    </CudaCompile>
";
    }
    print FLTS
"  </ItemGroup>
";
  }
  print FLTS
"</Project>
";
  close(FLTS);
}


sub ltrim {
   my $s = shift;
    $s =~ s/^\s+//;
     return $s;
}

sub isEmptyLine{
   my $line = shift;

   my $lineTrimmed  = ltrim($line);

   if($lineTrimmed eq "")
   {
       return 1;
   }
   else
   {
       return 0;
   }
}

sub isValidProjectLine{
   my $line = shift;

   if(!( isEmptyLine($line) || $line =~ m:(EXT_SUBDIRS_LIB.*\n{0,1}): ) )
   {
       return 1;
   }
   else
   {
       return 0;
   }
}




# ****************************************************
# ****************************************************
# ****************************************************

if (-e &{$osPathConversion}($solutionDir)) {
  print "Solution directory already exists, do you want to (r)emove, (o)verwrite, or (c)ancel? : ";
  my $ans = <STDIN>;
  if ($ans =~ /^c$/i) { exit 0; }
  elsif($ans =~ /^r/i) { &{$osPathConversion}($solutionDir); }
  elsif($ans !~ /^o/) { die "Invalid option given."; }
}
mkpath &{$osPathConversion}($solutionDir);
mkpath &{$osPathConversion}($projDir);

my $projguids = {};
loadHashTxtFile($projguidListFileName, $projguids);

my $projlist = {};
my $projdeps = {};
my $projlibs = {};

#my $makefiles = [];
#find(sub { if ($_ =~ /Makefile$/) { push @$makefiles, $File::Find::name; } },
#     &{$osPathConversion}($srcDir));

my @makefiles = (); # will be all the Makefiles in the subdirectories.
my $topLevelMakefile = "$srcDir/Makefile";
open(M, '<', &{$osPathConversion}($topLevelMakefile)) || die "opening $topLevelMakefile";
while(<M>) {
  # parsing the part of the top-level Makefile that's like:
  # SUBDIRS = base util matrix feat tree model fstext hmm optimization \
  #	    transform lm decoder bin fstbin gmmbin featbin
  if (s/^(SUBDIRS|EXT_SUBDIRS)\s+=\s+//) {
    # print STDERR "here\n";
    while ( isValidProjectLine($_) ) { # till we get an empty line or a line starting with EXT_SUBDIRS_LIB..
      s:\\::;
      foreach my $f (split(" ", $_)) {
       if($f eq "#"){
           last;
       }
        push @makefiles, "$srcDir/$f/Makefile";
      }
      $_ = <M>;
    }
  }
}
##foreach my $f (@makefiles) { print STDERR "Adding $f\n"; }

# was @$makefiles in the line below.
my $i = 0;
foreach my $makefile (@makefiles) {
  print "INFO: parsing " . $makefile . "\n";
  parseMakefile($makefile, $projlist, $projdeps, $projlibs);
	#print Dumper("Projlist", $projlist);
	#print Dumper("Projdeps", $projdeps);
	#print Dumper("Projlibs", $projlibs);
	#die "To staci" if $i >=3;
	$i++;
  }

# writeSolutionFile also creates guids for new projects
writeSolutionFile($solutionDir . '/' . $solutionFileName, $projlist, $projguids);

foreach my $propFile (@propsFiles) {
  copy(&{$osPathConversion}($propFile),
       &{$osPathConversion}($projDir . "/")) or die "ERROR: failed to copy prop file $propFile\n";
}
foreach my $option (keys %optionalProps) {
	if ($ENABLED{$option} ) {
		my $propFile = $optionalProps{$option};
		copy(&{$osPathConversion}($propFile), &{$osPathConversion}($projDir . "/")) or
			die "ERROR: failed to copy prop file $propFile\n";
	}
}

foreach my $projname (sort { $a cmp $b } keys %{$projlist->{ALL}}) {
  my $projFileDir = winPath(getProjFileDir($projname));
  mkpath &{$osPathConversion}($projFileDir);
  writeProjectFiles($projname, $projlist, $projdeps, $projlibs, $projguids);
}

saveHashTxtFile($projguidListFileName, $projguids);

# make Windows line-endings in non-Win environment
if ($^O !~ /MSWin32/i) {
  my $allfiles = [];
  find(sub { if ($_ =~ /sln$|vcxproj$|props$/) { push @$allfiles, $File::Find::name; } },
       &{$osPathConversion}($solutionDir));
  foreach my $file (@$allfiles) {
    checkCRLF($file);
  }
}
