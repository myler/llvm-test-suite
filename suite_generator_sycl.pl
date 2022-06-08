use strict;
use warnings;

BEGIN
{
    use lib $ENV{ICS_PERLLIB};
}
use icpuid;

use File::Basename;
use File::Spec;
use File::Copy;

use XML::Simple;
use Data::Dumper;
local $Data::Dumper::Sortkeys = 1;
local $Data::Dumper::Terse = 1;
local $Data::Dumper::Indent = 1;

my $command_status = 0;
my $command_output = "";

my $test_suite_repo = '.';
my $testbase = "/rdrive/tests/mainline/CT-SpecialTests/llvm-test-suite";
if (defined $ENV{TESTROOT}) {
  $testbase = "$ENV{TESTROOT}/CT-SpecialTests/llvm-test-suite";
}

my %feature_subsuite = ('SYCL' => {'Matrix'}, 'SYCL' => {'ESIMD'});

# HW/features depends on the output of 'tc show -gpu', or
# the feature list can be query by alloy API get_all_gpu_features();
# Supported HW/features are:
#   ats atsm dg1 dg2 double gen11 gen12 gen9 pvc subdevice
my @all_gpu_features = get_all_gpu_features();
# Reference: https://github.com/intel/llvm-test-suite/tree/intel/SYCL#lit-feature-checks
my $lit_feature_prefix = "gpu-intel-";

my $sycl_dir = '';

my $feature_folder = "SYCL";
my $feature_name = "";
my $config_folder = "config_sycl";
my $suite_name = "llvm_test_suite_sycl";
my $suite_description = "";
my $help_info = "Suite files generator for sycl tests\n\n"
              . "Usage: perl suite_generator_sycl.pl [tests_folder]\n"
              . "       perl suite_generator_sycl.pl tests_folder [description_file]\n\n"
              . "Argument explanation: \n"
              . "        - Empty, no argument. Generate tc files for SYCL folder(llvm_test_suite_sycl)\n"
              . "        - Argument [tests_folder] is the folder where you put your tests and\n"
              . "          the second argument [description_file] is the file which describes the suite.\n\n"
              . "          Notes: The folder name must be in uppercase and start with 'SYCL_'.\n\n"
              . "Examples:\n"
              . "          1)Generate tc files for folder SYCL_FEATURE_FOLDER\n"
              . "                perl suite_generator_sycl.pl SYCL_FEATURE_FOLDER\n"
              . "          2)Generate tc files for folder SYCL_FEATURE_FOLDER; and use the description in file DES.TXT\n"
              . "                perl suite_generator_sycl.pl SYCL_FEATURE_FOLDER DES.TXT\n\n";

sub append_file
{
  my $f = shift;

  my $fd;
  open $fd, ">>", $f or die "Failed to open for writing: $f: $!";
  # binmode as some our files have UTF-8 symbols
  binmode $fd; #, ':utf8';
  print $fd @_;
  close $fd or die "Failed to close after writing: $f: $!";
}

sub main
{
    $test_suite_repo = File::Spec->rel2abs($test_suite_repo);
    $sycl_dir = "$test_suite_repo/$feature_folder";

    execute("cd $sycl_dir && find -iname '*.cpp' | grep -vw 'Inputs' | sort");
    my @list = split( "\n", $command_output);
    execute("rm -rf $config_folder && mkdir $config_folder");

    my $tests = {};
    my %subsuite_tests;
    foreach my $t (@list)
    {
        my $path;
        if ( $t =~ /(.*)\.cpp$/) {
            $path = $1;
        } else {
            die "Wrong regexp";
        }

        $path =~ s/^\./$feature_folder/;
        my $name = $path;
        my $short_name = basename( $path);
        $path = dirname( $path);
        $name =~ s/$feature_folder\///;

        # Use $diff_name to save another name that is not duplicate with $name
        # $diff_name only replaces "/" with "_" and append "_0" in the end
        my $diff_name = $name;
        $diff_name =~ s/[\/]/_/g;
        $diff_name .= "_0";

        $name =~ s/[\/\-\.]/_/g;
        $name = lc $name;

        $name = $diff_name if defined $tests->{ $name};
        my $r = { name => $name, path => $path, fullpath =>"$path/$short_name.cpp", short_name => $short_name};

        $tests->{ $name} = $r;

        print( Dumper( $r));
        my $xml_text = gen_test( $r);
        print2file( $xml_text, "./$config_folder/$name.xml");

        # Save tests for subsuites
        foreach my $subsuite (sort keys %{ $feature_subsuite{$feature_folder} }) {
            if ($name =~ /^$subsuite/i) {
                $subsuite_tests{$subsuite}{$name} = $r;
            }
        }
    }

    print2file( gen_suite( $tests), "$suite_name.xml");
    print "\nThe number of tests in $suite_name: ";
    print scalar keys %{ $tests};
    # Print tests to suite xml file for subsuites
    foreach my $subsuite (sort keys %{ $feature_subsuite{$feature_folder} }) {
        my $subsuite_name = lc "${suite_name}_${subsuite}";
        print2file( gen_suite( $subsuite_tests{$subsuite}, $subsuite), "${subsuite_name}.xml");
        print "\nThe number of tests in $subsuite_name: ";
        print scalar keys %{ $subsuite_tests{$subsuite} };
    }
}

sub gen_suite
{
    my $tests = shift;
    my $subsuite = shift || "";
    ###
    my $xml = {};
    my $descr = "";
    my $current_suite_name = $suite_name;

    if ($subsuite ne "") {
        $current_suite_name = lc "${suite_name}_${subsuite}";
    }

    if ($suite_description ne "") {
        $descr = $suite_description;
    } else {
        $descr = "Port of $current_suite_name.\n"
               . "Suite is autogenerated by suite_generator_sycl.pl that you can find in the root dir of suite\n"
               . "Sources repo https://github.com/intel-innersource/applications.compilers.tests.llvm-project-llvm-test-suite\n";
    }

    $xml->{description} = { content => $descr};
    if ($feature_folder eq "SYCL") {
        if ($subsuite ne "") {
            # For subsuite
            $xml->{files}       = { file => [ { path => 'suite_generator_sycl.pl'}, { path => 'double_test.list'}, { path => 'cmake'}, { path => 'tools'}, { path => 'CMakeLists.txt'}, { path => 'litsupport'}, { path => 'lit.cfg'}, { path => 'lit.site.cfg.in'}, { path => 'SYCL/CMakeLists.txt', dst => 'SYCL/CMakeLists.txt'}, { path => 'SYCL/lit.cfg.py', dst => 'SYCL/lit.cfg.py'}, {path => 'SYCL/lit.site.cfg.py.in', dst => 'SYCL/lit.site.cfg.py.in'}, {path => 'SYCL/helpers.hpp', dst => 'SYCL/helpers.hpp'}, {path => 'SYCL/External/CMakeLists.txt', dst => 'SYCL/External/CMakeLists.txt'}, {path => 'SYCL/ExtraTests/CMakeLists.txt', dst => 'SYCL/ExtraTests/CMakeLists.txt'}, { path => "SYCL/${subsuite}", dst => "SYCL/${subsuite}"}, {path => 'SYCL/include', dst => 'SYCL/include'}, { path => '$INFO_TDRIVE/ref/lit'}, { path => $config_folder}, { path => '.github/CODEOWNERS'}]};
        } else {
            $xml->{rules} = { advancedRule => [{ perfSupport => 'accurate'}]};
            $xml->{files}       = { file => [ { path => 'suite_generator_sycl.pl'}, { path => 'double_test.list'}, { path => 'cmake'}, { path => 'tools'}, { path => 'CMakeLists.txt'}, { path => 'litsupport'}, { path => 'lit.cfg'}, { path => 'lit.site.cfg.in'}, { path => 'SYCL'}, { path => '$INFO_TDRIVE/ref/lit'}, { path => $config_folder}, { path => '.github/CODEOWNERS'}, { path => 'setenv.list'}]};
        }
    } else {
        if ($subsuite ne "") {
            # For subsuite
            $xml->{files}       = { file => [ { path => 'double_test.list'}, { path => 'cmake'}, { path => 'tools'}, { path => 'CMakeLists.txt'}, { path => 'litsupport'}, { path => 'lit.cfg'}, { path => 'lit.site.cfg.in'}, { path => 'SYCL'}, { path => "$feature_folder/$subsuite", dst => "SYCL_${subsuite}/$subsuite"}, { path => '$INFO_TDRIVE/ref/lit'}, { path => $config_folder}]};
        } else {
            $xml->{files}       = { file => [ { path => 'double_test.list'}, { path => 'cmake'}, { path => 'tools'}, { path => 'CMakeLists.txt'}, { path => 'litsupport'}, { path => 'lit.cfg'}, { path => 'lit.site.cfg.in'}, { path => 'SYCL'}, { path => $feature_folder}, { path => '$INFO_TDRIVE/ref/lit'}, { path => $config_folder}, { path => 'setenv.list'}]};
        }
    }

    my @strings = ();
    my $pre_xml_file = "${testbase}/$current_suite_name.xml";
    if ( -e $pre_xml_file ) {
        open my $fh, '<', $pre_xml_file or die "Could not open '$pre_xml_file'!\n";
        while (my $line = <$fh>) {
            chomp $line;
            push(@strings, $line)
        }
    }

    # Get tests that need double type support
    my $double_test_file = "double_test.list";
    my %double_test_list;
    if (-f $double_test_file) {
        my $double_tests = file2str($double_test_file);
        execute("sort -u double_test.list -o double_test.list");
        foreach my $test (split("^", $double_tests)) {
            $test =~ s/^\s+|\s+$//g;
            $double_test_list{$test} = 1;
        }
    }

    foreach my $testname ( sort keys %{ $tests})
    {
        my @pre_xml = ();
        my $pre_xml_name = "";

        my $group = "";
        if ( exists($double_test_list{$testname}) ) {
            $group = "double";
        } else {
            ( $group ) = split("_", lc($testname));
        }

        if ( @strings != 0 ) {
            @pre_xml = grep /testName="$testname"/, @strings;
        }
        if (-f "${config_folder}/$testname.xml") {
            push @{ $xml->{tests}{test}}, { configFile => "${config_folder}/$testname.xml", testName => $testname, splitGroup => $group};
            next;
        }
        if (@pre_xml != 0 and $pre_xml[0] =~ m/configFile="([^\s]*\.xml)"/) {
            $pre_xml_name = $1;
            my $pre_xml_file = basename($pre_xml_name);
            if (-f "${testbase}/$pre_xml_name" and ! -f "$pre_xml_file") {
                copy("${testbase}/$pre_xml_name", "./$config_folder/") or die "copy failed: $!";
                push @{ $xml->{tests}{test}}, { configFile => "${config_folder}/$testname.xml", testName => $testname, splitGroup => $group};
                next;
            }
        }
        # Suite and its subsuite use the same TEMPLATE xml file
        push @{ $xml->{tests}{test}}, { configFile => "${config_folder}/$testname.xml", testName => $testname, splitGroup => $group};
    }

    return XMLout( $xml, xmldecl => '<?xml version="1.0" encoding="UTF-8" ?>', RootName => 'suite');
}

sub token2feature
{
  my $r = shift;
  my $require_tokens = shift;
  my $unsupported_tokens = shift;

  my @tokens = ();

  # To the opposite to "// REQUIRES:" rule.
  # But with higher priority, e.g.
  # A test with following two lines:
  # // REQUIRES: pvc
  # // UNSUPPORTED: pvc
  # Then the test would be skipped in pvc.
  foreach my $ut (@{$unsupported_tokens}) {
    next if (not defined $ut or $ut eq "");
    if (substr($ut, 0, 1) eq '!') {
      my $subtoken = substr($ut, 1);
      foreach my $feature (@all_gpu_features) {
        if ($subtoken eq "$lit_feature_prefix$feature" or $subtoken eq "aspect-fp64" and $feature eq "double") {
          push(@tokens, "!$feature");
          last;
        }
      }
    } else {
      foreach my $feature (@all_gpu_features) {
        if ($ut eq "$lit_feature_prefix$feature" or $ut eq "aspect-fp64" and $feature eq "double") {
          push(@tokens, $feature);
          last;
        }
      }
    }
  }

  # the main rule is to exclude running some unwanted test.
  # so a require is met, that is to say we won't skip exclude it, so '!' prefix is required.
  foreach my $rt (@{$require_tokens}) {
    next if (not defined $rt or $rt eq "");
    if (substr($rt, 0, 1) eq '!') {
      my $subtoken = substr($rt, 1);
      foreach my $feature (@all_gpu_features) {
        if ($subtoken eq "$lit_feature_prefix$feature"or $subtoken eq "aspect-fp64" and $feature eq "double") {
          my $seen = 0;
          foreach my $ref (@tokens) {
            my $ref_token = $ref;
            $ref_token = substr($ref, 1) if (substr($ref, 0, 1) eq '!');
            if ($subtoken eq $ref_token) {
              $seen = 1;
              last;
            }
          }
          push(@tokens, $feature) if (not $seen);
          last;
        }
      }
    } else {
      foreach my $feature (@all_gpu_features) {
      #if (grep(/\Q$lit_feature_prefix$rt/, @all_gpu_features)) {
        if ($rt eq "$lit_feature_prefix$feature" or $rt eq "aspect-fp64" and $feature eq "double") {
          my $seen = 0;
          foreach my $ref (@tokens) {
            my $ref_token = $ref;
            $ref_token = substr($ref, 1) if (substr($ref, 0, 1) eq '!');
            if ($rt eq $ref_token) {
              $seen = 1;
              last;
            }
          }
          push(@tokens, "!$feature") if (not $seen);
        }
      }
    }
  }

  # Remove duplicated tokens.
  my %hash_tmp = map { $_, 1 } @tokens;
  @tokens = keys %hash_tmp;

  foreach my $token (@tokens) {
    # special handler for specified tokens.
    if ($token eq "!double") {
      append_file("double_test.list", "$r->{name}\n") if (-f "double_test.list");
    }
  }

  return @tokens;
}

sub feature2rule
{
  my $xml_ref = shift;
  my $features_ref = shift;

  if (scalar @${features_ref}) {
    $$xml_ref->{rules} = { optlevelRule => []};
  }

  foreach my $feature (@{$features_ref}) {
    push(@{$$xml_ref->{rules}{optlevelRule}}, { GPUFeature => "$feature", excludeOptlevelNameString => 'gpu'});
  }
}

sub translate_gpu_feature
{
  my $r = shift;
  my $xml_ref = shift;
  my $f = $r->{fullpath};
  my @gpu_features = ();

  my $lines = file2str($f);
  my @require_tokens = ();
  my @unsupported_tokens = ();
  foreach my $line (split /^/, $lines) {
    # Only ' ', ','  and ' ,' are the separate delimilers.
    # '!' is supported to reverse a require, e.g. "!pvc" means the test is skipped on pvc.
    # Not support "AND", "OR", "()"
    if ($line =~ /\/\/ REQUIRES: *(.*)$/) {
      my $token_str = $1;
      @require_tokens = split(/, |,| \&\& |\&\&| \|\| |\|\|/, $token_str);
    } elsif ($line =~ /\/\/ UNSUPPORTED: *(.*)$/) {
      my $token_str = $1;
      @unsupported_tokens = split(/, |,| \&\& |\&\&| \|\| |\|\|/, $token_str);
    }
  }

  @gpu_features = token2feature($r, \@require_tokens, \@unsupported_tokens);
  feature2rule($xml_ref, \@gpu_features);
}

sub requires2rule
{
  my $r = shift;
  my $xml_ref = shift;

  translate_gpu_feature($r, $xml_ref);
}

sub gen_test
{
    my $r = shift;
    my $f = $r->{fullpath};
    my $xml = {};
    $xml->{driverID} = 'llvm_test_suite_sycl';
    $xml->{name}     = "$r->{name}";
    $xml->{description} = { content => "WARNING: DON'T UPDATE THIS FILE MANUALLY!!!\nThis config file auto-generated by suite_generator_sycl.pl."};

    requires2rule($r, \$xml);

    print2file( "$r->{path}/$r->{short_name}.cpp", "./$config_folder/$r->{name}.info");

    return XMLout( $xml, xmldecl => '<?xml version="1.0" encoding="UTF-8" ?>', RootName => 'test');
}

sub file2str
{
    my $file = shift;
    ###
    local $/=undef;
    open FD, "<$file" or die "Fail to open file $file!\n";
    binmode FD;
    my $str = <FD>;
    close FD;
    return $str;
}

sub print2file
{
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">$file";

    print FD $s;
    close FD;
}

sub dump2file
{
    my $r = shift;
    my $file = shift;
    ###
    print2file( Dumper( $r), $file);
}

sub execute
{
    my $cmd = shift;
    ###

    print "$cmd\n";
    $command_output = `$cmd 2>&1`;
    my $code = $?;
    my $perl_err        = $code & ( ( 1 << 8) - 1);
    my $shell_err       = $code >> 8;

    $command_status = $shell_err;

    return ( $command_status, $command_output);
}

if ( scalar(@ARGV) == 0 ) {
    print "Generate tc files only for llvm_test_suite_sycl\n\n";
} elsif ( scalar(@ARGV) == 1 or scalar(@ARGV) == 2 ) {
    if ($ARGV[0] =~ /^-h/i or $ARGV[0] =~ /^--help/i) {
        print "$help_info";
        exit 0;
    }

    $feature_folder = $ARGV[0];
    $feature_folder =~ s/\/$//;
    $feature_folder = basename($feature_folder);
    if ( $feature_folder ne 'SYCL' and $feature_folder !~ /^SYCL_/ ) {
        die "Unsupported folder $feature_folder! Please make sure the folder name is 'SYCL' or starts with 'SYCL_'.\n\n$help_info";
    }

    if ( defined $ARGV[1] ) {
        my $description_file =  $ARGV[1];
        if ( -f $description_file) {
            $suite_description = file2str($description_file);
        } else {
            die "File $description_file doesn't exist!\n\n";
        }
    }

    $feature_name = $feature_folder;
    $feature_name =~ s/^SYCL_//;
    $feature_name = lc $feature_name;
    if ( $feature_folder ne 'SYCL' ) {
        $config_folder = $config_folder . '_' . $feature_name;
    }
    $suite_name = "llvm_test_suite_" . $feature_name;
    print "Generate tc files for $suite_name\n\n";
} else {
    die "Error: The number of arguments is larger than 2!\n\n$help_info";
}

main();
print "\n\nFinish the generation of $suite_name Successfully.\n";
if ( $suite_name eq "llvm_test_suite_sycl" ) {
    my $valgrind_suite_name = "llvm_test_suite_sycl_valgrind";
    copy("$suite_name.xml", "$valgrind_suite_name.xml");
    print "\n\nFinish the generation of $valgrind_suite_name Successfully.\n";
}


