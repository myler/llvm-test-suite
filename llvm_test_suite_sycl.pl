use File::Basename;
use File::Copy;

my $cmake_log = "$optset_work_dir/cmake.log";
my $cmake_err = "$optset_work_dir/cmake.err";
# Use all.lf to store standard output of run.
my $run_all_lf = "$optset_work_dir/run_all.lf";

my $is_dynamic_suite = 0;

# @test_to_run_list stores only the test(s) that will be run
# For example, for "tc -t llvm_test_suite_sycl/aot_cpu,aot_gpu" it will store 2 tests - aot_cpu and aot_gpu
my @test_to_run_list = get_tests_to_run();
# @suite_test_list stores all the tests in the whole suite(without splitting) or sub-suite(with splitting)
# For example, for "tc -t llvm_test_suite_sycl~4-1/aot_cpu,aot_gpu" it will store all the tests in sub-suite 4-1
my @suite_test_list = get_test_list($current_optset);
my $short_test_name;
my $test_info;
my $config_folder = 'config_sycl';
my $subdir = "SYCL";
my $insert_command = "";
my $valgrind_dir = "$optset_work_dir/_VALGRIND/valgrind_reports";

my $sycl_backend = "";
my $device = "";
my $os_platform = is_windows() ? "windows" : "linux";

my $build_dir = "$optset_work_dir/build";
my $lit = "../lit/lit.py";

sub gpu {
  my $gpus = shift;
  $gpus = [ $gpus ] if ref($gpus) ne "ARRAY";
  my $current_gpu = $ENV{'CURRENT_GPU_DEVICE'};
  if (!defined $current_gpu) {
    return 0;
  } else {
    $current_gpu =~ tr/,//d; # e.g. gen9,/gen9/double/,GEN9
    if (grep(/$current_gpu/i, @{$gpus})) {
      return 1;
    } else {
      return 0;
    }
  }
}

sub lscl {
    my $args = shift;

    my $os_flavor = is_windows() ? "win.x64" : "lin.x64";

    my $lscl_bin = $ENV{"ICS_TESTDATA"} .
      "/mainline/CT-SpecialTests/opencl/tools/$os_flavor/bin/lscl";

    my @cmd = ($lscl_bin);

    push(@cmd, @{$args}) if $args;

    # lscl show warning "clGetDeviceInfo failed: Invalid value"
    # which contains string "fail", cause tc fail
    # Remove OS type check because RHEL8 has the same issue
    push(@cmd, "--quiet");

    execute(join(" ", @cmd));

    my $output = "\n  ------ lscl output ------\n"
               . "$command_output\n";

    return $output;
}

sub get_dynamic_test_list
{
    @list = map { s/^.*$config_folder\///; s/\.info//; $_ } alloy_find("$optset_work_dir/$config_folder", '.*\.info');
    #log_command("##test list: @list");

    return @list;
}

sub is_zperf_run {
    if ((defined $opt_perf && $opt_perf) ||
        (defined $opt_perf_run && $opt_perf_run)) {
        return 1;
    }
    return 0;
}

sub is_suite {
    my @whole_suite_test = sort(@suite_test_list);
    my @current_test_list = sort(@test_to_run_list);

    return is_same(\@current_test_list, \@whole_suite_test);
}

sub init_test
{
    if ($current_suite =~ /compatibility_llvm_test_suite_sycl/) {
      my @folder_list = ("SYCL", $config_folder);
      my $folder_not_exist = 0;
      my $sparse_file_in_git = ".git/info/sparse-checkout";
      my $sparse_co_file = "$optset_work_dir/sparse-checkout";
      print2file("# Use sparse-checkout to get files\n", $sparse_co_file);
      foreach my $folder (@folder_list) {
        append2file("$folder\n", $sparse_co_file);
        if (! -d "$optset_work_dir/$folder") {
          $folder_not_exist = 1;
        }
      }

      if ($folder_not_exist == 1) {
        my $compiler_path = "";
        if (defined $ENV{BASECOMPILER}) {
          $compiler_path = $ENV{BASECOMPILER};
        } else {
          $compiler_path = generate_tool_path("dpcpp", "clang++");
        }

        my $branch = "";
        my $date = "";
        if ( $compiler_path =~ /deploy_(xmain-rel)\/xmainefi2[a-z]{1,}\/([0-9]{4})([0-9]{2})([0-9]{2})_[0-9]{6}/) {
          $branch = $1;
          $date = "$2-$3-$4";
          log_command("##Branch: $branch, Date: $date");
        } else {
          $failure_message = "fail to get date of base compiler";
          return $COMPFAIL;
        }

        my $git_repo = unxpath("$ENV{ICS_GIT_MIRROR}/applications.compilers.tests.llvm-project-llvm-test-suite.git");

        log_command("##Clone repo");
        #copy($sparse_co_file, $sparse_file_in_git);
        rmtree(".git"); # Clean .git folder
        my @get_repo_cmds = ();
        push(@get_repo_cmds, "git init");
        push(@get_repo_cmds, "git config core.sparseCheckout true");
        push(@get_repo_cmds, "cp $sparse_co_file $sparse_file_in_git");
        push(@get_repo_cmds, "git remote add -t ${branch} -f origin ${git_repo}");
        push(@get_repo_cmds, "git checkout ${branch}");

        foreach my $cmd (@get_repo_cmds) {
          execute($cmd);
          $compiler_output .= $command_output;
          if ($command_status != 0) {
            $failure_message = "fail to clone repo";
            return $COMPFAIL;
          }
        }

        log_command("##Get hash from bare repo");
        my $get_hash_cmd = "git log -1 --before=\"$date\" --pretty=format:\"%h\" --first-parent";
        execute($get_hash_cmd);
        $compiler_output .= $command_output;
        if ($command_status != 0) {
          $failure_message = "fail to get hash before $date in $branch branch";
          return $COMPFAIL;
        }
        my $hash = $command_output;
        log_command("##hash: $hash");

        my $archive_cmd = "git checkout $hash";
        execute($archive_cmd);
        $compiler_output .= $command_output;
        if ($command_status != 0) {
          $failure_message = "fail to get folder(s)";
          return $COMPFAIL;
        }
      } else {
        log_command("##Reuse folder(s)");
      }

      # Get test list
      @test_to_run_list = get_dynamic_test_list();

      return $PASS;
    }

    my $suite_feature = $current_suite;
    $suite_feature =~ s/^llvm_test_suite_//;
    #Remove suffix of suite names if it has
    $suite_feature =~ s/~.*$//;
    if ($suite_feature !~ /^sycl/)
    {
      $config_folder = $config_folder . '_' . $suite_feature;
      $subdir = uc $suite_feature;
      $subdir = 'SYCL_' . $subdir;

      my $sycl_dir = "./SYCL";
      my @file_list = alloy_find($sycl_dir, '(.*\.(h|hpp|H|HPP)|lit\..*|CMakeLists\.txt)');

      # Copy files to folder $subdir
      foreach my $file (@file_list) {
        $file =~ s/^\.\/SYCL\///;
        my $rel_file_path = dirname($file);
        my $file_path_in_subdir = "$optset_work_dir/$subdir";
        if ($rel_file_path ne '.') {
          $file_path_in_subdir = $file_path_in_subdir . "/$rel_file_path";
        }
        my $file_in_sycl = "$optset_work_dir/SYCL/$file";
        if ( -d $file_path_in_subdir) {
          cp($file_in_sycl, $file_path_in_subdir);
        }
      }

      my $cmake_file = "$subdir/CMakeLists.txt";
      if ( ! -d "$subdir/External") {
        `sed -i '/^add_subdirectory(External)/s/^/#/g' $cmake_file`;
      }
      if ( ! -d "$subdir/ExtraTests"){
        `sed -i '/^add_subdirectory(ExtraTests)/s/^/#/g' $cmake_file`;
      }
    }

    #Remove untested source files from $subdir if it run with several subsuites
    if (is_suite()) {
      my $info_dir = "$optset_work_dir/$config_folder";
      my @info_files = glob("$info_dir/*.info");

      my %in_test_hash = map { $_ => 1 } @suite_test_list;
      my @outof_test_list = ();

      for my $file (@info_files) {
        $file = basename($file);
        $file =~ s/\.info//;
        if (!exists($in_test_hash{$file})) {
          push(@outof_test_list, $file);
        }
      }
      for my $test (@outof_test_list) {
        my $test_info = get_info($test);
        my $path = "$test_info->{fullpath}";
        rmtree($path);
      }
      log_command("##Removed tests that are not in $current_suite\n");
    }

    if ($current_suite =~ /valgrind/) {
      safe_Mkdir('-p',$valgrind_dir);
      $insert_command = "$ENV{INFO_RDRIVE}/ref/valgrind/v3.16.0/efi2/bin/valgrind --leak-check=full --show-leak-kinds=all --trace-children=yes --log-file=$valgrind_dir/v.%basename_t.%%p.log";
    }

    return PASS;
}

sub extract_perf_results
{
    my $timer = Timer->new($current_test, $current_suite, $current_optset);
    $timer->set("host", &alloy_utils::get_hostname());
    my $output_file = join($slash, $optset_work_dir, "$current_test.output");
    open(LOG, "+>", $output_file) or die "open $output_file fail";
    print LOG $execution_output;
    seek(LOG, 0, 0);
    my $perf_matched = 0;
    while (<LOG>) {
        my $pattern = ".*OverallTime(.*):(\\d+.?\\d*[Ee]?[+-]?\\d+).*";
        if ($_ =~ qr/$pattern/) {
            my $primary = $1;
            my $result = $2;
            my $metric = "time";
            my $better = "lt";
            $timer->set($metric, $result);
            $timer->set("BETTER_$metric", $better);
            if ($primary =~ m/Primary/) {
              $timer->set("primary_metric", $metric);
              $perf_matched = 1;
            }
        }
        $pattern = ".*KernelThroughput(.*):(\\d+.?\\d*[Ee]?[+-]?\\d+).*";
        if ($_ =~ qr/$pattern/) {
            my $primary = $1;
            my $result = $2;
            my $metric = "throughput";
            my $better = "gt";
            $timer->set($metric, $result);
            $timer->set("BETTER_$metric", $better);
            if ($primary =~ m/Primary/) {
              $timer->set("primary_metric", $metric);
              $perf_matched = 1;
            }
        }
        $pattern = ".*KernelTime(.*):(\\d+.?\\d*[Ee]?[+-]?\\d+).*";
        if ($_ =~ qr/$pattern/) {
            my $primary = $1;
            my $result = $2;
            my $metric = "kerneltime";
            my $better = "lt";
            $timer->set($metric, $result);
            $timer->set("BETTER_$metric", $better);
            if ($primary =~ m/Primary/) {
              $timer->set("primary_metric", $metric);
              $perf_matched = 1;
            }
        }
    }
    close(LOG);
    if (! $perf_matched) {
        print "Warning: Primary metric is not specified!!!\n";
    }
}

sub init_and_cmake
{
    my $init_status = init_test();
    if ($init_status != $PASS) {
        return $init_status;
    }
    log_command("##Finish getting source code");

    # Before running cmake, add settings for specific tests according to setenv.list
    add_setting();

    my ( $status, $output) = run_cmake();
    if ( $status)
    {
        log_command("##Fail in cmake. Rename $cmake_log to $cmake_err.");
        rename($cmake_log, $cmake_err);
    } else {
        # If there is no configuration issue, print device info
        my $lscl_output = lscl();
        append2file($lscl_output, $cmake_log);
    }
    return $PASS;
}

sub run_and_parse
{
    if ( -f $cmake_log)
    {
        $compiler_output = file2str($cmake_log);
        $test_info = get_info();
        my ( $status, $output) = do_run($test_info);
        my $res = "";
        if (-e $run_all_lf)
        {
            my $run_output = file2str("$run_all_lf");
            $res = generate_run_result($run_output);
            my $filtered_output = generate_run_test_lf($run_output);
            $execution_output .= $filtered_output;
        } else {
            $res = generate_run_result($output);
        }
        if ($res eq $PASS && is_zperf_run()) {
            extract_perf_results();
        }
        if ($current_suite =~ /valgrind/) {
            my $test_basename = $test_info->{"short_name"};
            my @log_list = alloy_find($valgrind_dir, "v\.$test_basename\.[0-9]{1,}\.log");
            if ( scalar(@log_list) > 0 ) {
              $execution_output .= "\nVALGRIND reports problems. Check the following log files for detailed report:\n";
              foreach my $log (@log_list) {
                $execution_output .= "$log\n";
              }

              my $scrdir = $ENV{TC_MEMCHECK_SCRIPTDIR} ||
                           "$ENV{ICS_PKG_QATOOLS}/valgrind_tool";
              if ( -f "$scrdir/process_logs_for_TC.pm") {
                push @INC, $scrdir;
                require process_logs_for_TC;
                import process_logs_for_TC;

                $execution_output .= "\nProcess valgrind logs by process_logs_for_TC.pm\n";
                # Save compilation and execution output because process_logs will overwrite it
                my $compiler_output_ori = $compiler_output;
                my $execution_output_ori = $execution_output;
                $compiler_output = '';
                $execution_output = '';
                process_logs(\&finalize_test, $valgrind_dir, $test_basename, $RUNFAIL);
                # Recover compilation and execution output
                $execution_output = $execution_output_ori;
                $compiler_output = $compiler_output_ori;
              }

              $failure_message = "VALGRIND reports problems. Original result: ";
              if ($res eq $PASS) {
                $failure_message .= "passed";
              } else {
                $failure_message .= "failed";
              }
              return $RUNFAIL;
            }
        }
        return $res;
    } elsif ( -f $cmake_err) {
        $compiler_output = file2str($cmake_err);
    }

    $failure_message = "cmake returned non zero exit code" if ($failure_message eq "");
    return $COMPFAIL;
}

sub BuildSuite
{
    my $status = init_and_cmake();
    if ($status != $PASS) {
        log_command("##Fail in init and/or cmake stage.");
        # Report testing result
        report_result("ALL", $status, $failure_message, $compiler_output, "");
        return $status;
    }
    return $PASS;
}

sub RunSuite
{
    $is_dynamic_suite = 1;

    my $run_status = $PASS;
    foreach my $test (@test_to_run_list) {
        $current_test = $test;
        my $status = run_and_parse();
        if ($status == $SKIP or $status == $PASS) {
            $failure_message = "";
        } else {
            $run_status = $RUNFAIL;
        }
        # Report testing result
        report_result($test, $status, $failure_message, $compiler_output, $execution_output);
    }
    return $run_status;
}

sub BuildTest
{
    return 0;
}

sub RunTest
{
    $is_dynamic_suite = 0;
    if ($current_test eq $test_to_run_list[0])
    {
        init_and_cmake();
    } else {
        chdir_log($build_dir);
    }
    return run_and_parse();
}

sub modify_test_file
{
    my $test = shift;
    my $placeholder = shift;
    my $setting = shift;

    my $test_info = get_info($test);
    my $test_file = $test_info->{fullpath};
    # Use absolute path for test
    $test_file = "$optset_work_dir/$test_file";
    if (-f $test_file) {
        log_command("##Add \"$setting\" to \"$placeholder\" for test $test\n");
        my $test_file_original = "${test_file}.ori";
        copy($test_file, $test_file_original);

        open my $in, "<", $test_file_original || die "Cannot open file $test_file_original: $!";
        open my $out, ">", $test_file || die "Cannot open file $test_file: $!";
        while (<$in>) {
          s/($placeholder)/$1 $setting /g;
          print $out $_;
        }
        close $in;
        close $out;
    } else {
        die "Cannot find file $test_file: $!";
    }
}

sub add_setting
{
    my $list_file = "$optset_work_dir/setenv.list";
    if (! -f $list_file) {
        log_command("##setenv.list doesn't exist so no extra setting is added\n");
        return;
    }

    my $rules = file2str($list_file);
    foreach my $rule (split(/^/, $rules)) {
        if ($rule =~ /^#/) {
            next;
        }
        if ($rule =~ /([^,]{1,}),([^,]{1,}),([^,]{0,}),([^,]{1,})/) {
            my $test_pattern = $1;
            my $optset_pattern = $2;
            my $platform_pattern = $3;
            my $setting = $4;
            if ($current_optset !~ m/$optset_pattern/) {
                # optset is not matched
                # log_command("##optset $current_optset is not matched with $optset_pattern\n");
                next;
            } elsif ($platform_pattern ne "" and $os_platform !~ m/$platform_pattern/) {
                # platform is not matched
                next;
            } else {
                # check whether test is matched
                my @match_tests = grep { /$test_pattern/ } @test_to_run_list;
                if (scalar(@match_tests) == 0) {
                    # test is not matched
                    next;
                } else {
                    # test is matched
                    if ($current_optset =~ /opt_use_(cpu|acc|gpu|host)/) {
                        my $device = uc($1);
                        my $placeholder = "%${device}_RUN_PLACEHOLDER";
                        # log_command("##optset $current_optset placeholder $placeholder\n");
                        foreach my $test (@match_tests) {
                            $setting =~ s/\s+$//;
                            modify_test_file($test, $placeholder, $setting);
                        }
                    } else {
                        die "Unrecognized optset: $optset_pattern!\n";
                    }
                }
            }
        } else {
            die "Unrecognized format: $rule!\n";
        }
    }
}

sub do_run
{
    my $r = shift;
    my $path = "$r->{fullpath}";

    if (! -e $run_all_lf) {
      my $python = "python3";
      my $timeset = "";
      # Set matrix to 1 if it's running on ATS or using SPR SDE
      my $matrix = "";
      my $jobset = "-j 8";
      my $zedebug = "";
      my $gpu_opts = "";

      if ( is_ats() ) {
        $python = "/usr/bin/python3";
        $matrix = "-Dmatrix=1";
        $jobset = "-j 1";
      } elsif ( is_pvc() ) {
        $matrix = "-Dmatrix-pvc=1";
        $timeset = "--timeout 1800";
        $jobset = "";
      }

      if ($current_optset =~ m/_spr$/) {
        $matrix = "-Dmatrix=1";
      }

      if ($current_optset =~ m/_zedebug/) {
        $zedebug = "--param ze_debug=-1";
      }

      if ($current_suite =~ m/valgrind/){
        $timeset = "--timeout 0";
      }

      if (gpu(['dg1', 'dg2', 'ats', 'pvc'])) {
        $gpu_opts .= "-Dgpu-intel-dg1=1";
      }

      set_tool_path();
      if ($is_dynamic_suite == 1 or is_suite()) {
        execute("$python $lit -a $gpu_opts $matrix $zedebug $jobset . $timeset > $run_all_lf 2>&1");
      } else {
        execute("$python $lit -a $gpu_opts $matrix $zedebug $path $timeset");
      }
    }

    $execution_output = "$command_output";
    return $command_status, $command_output;
}

sub set_tool_path
{
    my $tool_path = "";
    if ($cmplr_platform{OSFamily} eq "Windows") {
        $tool_path = "$optset_work_dir/lit/tools/Windows";
    } else {
        $tool_path = "$optset_work_dir/lit/tools/Linux";
    }
    my $env_path = join($path_sep, $tool_path, $ENV{PATH});
    set_envvar("PATH", $env_path, join($path_sep, $tool_path, '$PATH'));

    # For the product compiler, add the internal "bin-llvm" directory to PATH.
    if ($compiler =~ /xmain/) {
        my $llvm_dir = dirname(qx/dpcpp -print-prog-name=llvm-ar/);
        my $llvm_path = join($path_sep, $llvm_dir, $ENV{PATH});
        set_envvar("PATH", $llvm_path, join($path_sep, $llvm_dir, '$PATH'));
    }

}

sub get_info
{
    my $test_name = shift;
    $test_name = $current_test if ! defined $test_name or $test_name eq "";

    my $test_file = file2str("$optset_work_dir/$config_folder/$test_name.info");
    $short_test_name = $test_file;
    $short_test_name =~ s/^$subdir\///;

    my $short_name = basename($test_file);
    my $path = dirname($test_file);
    my $r = { dir => $path, short_name => $short_name, fullpath => $test_file};

    return $r;
}

sub generate_run_result
{
    my $output = shift;
    my $result = "";
    for my $line (split /^/, $output){
      if ($line =~ m/^(.*): SYCL :: \Q$short_test_name\E \(.*\)/) {
        $result = $1;
        if ($result =~ m/^PASS/ or $result =~ m/^XFAIL/) {
          # Expected PASS and Expected FAIL
          return $PASS;
        } elsif ($result =~ m/^XPASS/) {
          # Unexpected PASS
          $failure_message = "Unexpected pass";
          return $RUNFAIL;
        } elsif ($result =~ m/^TIMEOUT/) {
          # Exceed test time limit
          $failure_message = "Reached timeout";
          return $RUNFAIL;
        } elsif ($result =~ m/^FAIL/) {
          # Unexpected FAIL
          next;
        } elsif ($result =~ m/^UNSUPPORTED/) {
          # Unsupported tests
          return $SKIP;
        } else {
          # Every test should have result.
          # If not, it is maybe something wrong in processing result or missing result
          $failure_message = "Result not found";
          return $FILTERFAIL;
        }
      }

      if ($result =~ m/^FAIL/) {
        if ($line =~ m/Assertion .* failed/ or $line =~ m/Assertion failed:/) {
          $failure_message = "Assertion failed";
          return $RUNFAIL;
        } elsif ($line =~ m/No device of requested type available/) {
          $failure_message = "No device of requested type available";
          return $RUNFAIL;
        } elsif ($line =~ m/error: CHECK.*: .*/) {
          $failure_message = "Check failed";
          return $RUNFAIL;
        } elsif ($line =~ m/fatal error:.* file not found/) {
          $failure_message = "File not found";
          return $RUNFAIL;
        } elsif ($line =~ m/error: command failed with exit status: ([\-]{0,1}[0]{0,1}[x]{0,1}[0-9a-f]{1,})/) {
          $failure_message = "command failed with exit status $1";
          return $RUNFAIL;
        }
      }
    }

    # Every test should have result.
    # If not, it is maybe something wrong in processing result or missing result
    $failure_message = "Result not found";
    return $FILTERFAIL;
}

sub generate_run_test_lf
{
    my $output = shift;
    my $filtered_output = "";

    my $printable = 0;
    for my $line (split /^/, $output) {
      if ($line =~ m/^.*: SYCL :: \Q$short_test_name\E \(.*\)/) {
        $printable = 1;
        $filtered_output .= $line;
        next;
      }

      if ($printable == 1) {
        if ($line =~ m/^[*]{20}/ and length($line) <= 22) {
          $filtered_output .= $line;
          $printable = 0;
          last;
        } else {
          $filtered_output .= $line;
        }
      }
    }
    return $filtered_output;
}

# Call a driver to obtain a path to a particular tool. On Windows, backslashes
# are converted to forward slashes and ".exe" is appended such that CMake will
# accept the string as a compiler name.
sub generate_tool_path
{
    my $driver = shift;
    my $tool_name = shift;

    my $tool_path = qx/$driver --print-prog-name=$tool_name/;
    chomp $tool_path;

    if ($cmplr_platform{OSFamily} eq "Windows") {
        $tool_path =~ tr#\\#/#;
        $tool_path = "$tool_path.exe";
    }

    return $tool_path;
}

sub run_cmake
{
    my $c_flags = "$current_optset_opts $compiler_list_options $compiler_list_options_c $opt_c_compiler_flags";
    my $cpp_flags = "$current_optset_opts $compiler_list_options $compiler_list_options_cpp $opt_cpp_compiler_flags";
    my $link_flags = "$linker_list_options $opt_linker_flags";
    my $c_cmplr = &get_cmplr_cmd('c_compiler');
    my $cpp_cmplr = &get_cmplr_cmd('cpp_compiler');
    my $c_cmd_opts = '';
    my $cpp_cmd_opts = '';
    my $thread_opts = '';
    my $gpu_aot_target_opts = '';

    ($c_cmplr, $c_cmd_opts) = remove_opt($c_cmplr);
    ($cpp_cmplr, $cpp_cmd_opts) = remove_opt($cpp_cmplr);
    $c_cmd_opts .= $c_flags;
    $cpp_cmd_opts .= $cpp_flags;

    if ($cmplr_platform{OSFamily} eq "Windows") {
    # Windows
        if ($compiler !~ /xmain/) {
            $c_cmplr = "clang-cl";
            $cpp_cmplr = "clang-cl";
            # Add "/EHsc" for syclos
            $cpp_cmd_opts .= " /EHsc";
        } else {
            $c_cmplr = "clang";
            $cpp_cmplr = 'clang++';
            # Clang is not guaranteed to be in PATH, but dpcpp is. Ask dpcpp
            # for absolute paths.
            $c_cmplr = generate_tool_path("dpcpp", $c_cmplr);
            $cpp_cmplr = generate_tool_path("dpcpp", $cpp_cmplr);
            $c_cmd_opts = convert_opt($c_cmd_opts);
            $cpp_cmd_opts = convert_opt($cpp_cmd_opts);
        }
    } else {
    # Linux
        $c_cmplr = "clang";
        if ($compiler =~ /xmain/) {
            $cpp_cmplr = "clang++";
            # Clang is not guaranteed to be in PATH, but dpcpp is. Ask dpcpp
            # for absolute paths.
            $c_cmplr = generate_tool_path("dpcpp", $c_cmplr);
            $cpp_cmplr = generate_tool_path("dpcpp", $cpp_cmplr);
        }
        $thread_opts = "-lpthread";
    }

    my $collect_code_size="Off";
    execute("which llvm-size");
    if ($command_status == 0)
    {
        $collect_code_size="On";
    }

    if ( $current_optset =~ m/ocl/ )
    {
        $sycl_backend = "PI_OPENCL";
    } elsif ( $current_optset =~ m/nv_gpu/ ) {
        $sycl_backend = "PI_CUDA";
    } elsif ( $current_optset =~ m/gpu/ ) {
        $sycl_backend = "PI_LEVEL_ZERO";
    } else {
        $sycl_backend = "PI_OPENCL";
    }

    if ( $current_optset =~ m/opt_use_cpu/ )
    {
        $device = "cpu";
    }elsif ( $current_optset =~ m/opt_use_gpu/ ){
        $device = "gpu";
        if ( is_pvc() ) {
          execute("lspci | grep Display");
          if( $command_status == 0 and $command_output =~ /([0-9a-f]{4}) \(rev ([0-9]{1,})\)/i ) {
            my $device_id = $1;
            my $device_rev = $2;
            $gpu_aot_target_opts = "-DGPU_AOT_TARGET_OPTS=\"\\\'-device 0x${device_id} -revision_id ${device_rev}\\\'\"";
          } else {
            log_command("##Warning: Fail to get device and revision id!");
          }
        }
    }elsif ( $current_optset =~ m/opt_use_acc/ ){
        $device = "acc";
    }elsif ( $current_optset =~ m/opt_use_nv_gpu/ ){
        $device = "gpu";
    }else{
        $device = "host";
    }

    my $lit_extra_env = "SYCL_ENABLE_HOST_DEVICE=1";
    $lit_extra_env = join_extra_env($lit_extra_env,"GCOV_PREFIX");
    $lit_extra_env = join_extra_env($lit_extra_env,"GCOV_PREFIX_STRIP");
    $lit_extra_env = join_extra_env($lit_extra_env,"TC_WRAPPER_PATH");
    $lit_extra_env = join_extra_env($lit_extra_env,"TBB_DLL_PATH");
    $lit_extra_env = join_extra_env($lit_extra_env,"ZE_AFFINITY_MASK");

    if ( defined $ENV{PIN_CMD} ) {
        my $pin_cmd = $ENV{PIN_CMD};

        # Only pass PIN_CMD to lit when PIN_CMD includes "=" and does not include " "(space)
        if ($pin_cmd =~ /=(.*)/) {
          my $pin_cmd_value = $1;
          $pin_cmd_value =~ s/\s+$//; # Remove the ending space
          if ($pin_cmd_value !~ / /) {
            $lit_extra_env = join(',',$lit_extra_env,$ENV{PIN_CMD});
          }
        } elsif ($insert_command eq "") {
          $insert_command = $pin_cmd;
        }
    }

    if ($insert_command ne "") {
        my $config_file = "$optset_work_dir/SYCL/lit.cfg.py";
        if (! -f $config_file) {
          return COMPFAIL, "File SYCL/lit.cfg.py doesn't exist";
        }

        my $config_file_original = "$config_file.ori";
        # If using tc -rerun, it may repeat inserting so we need to keep the original file and insert on it
        if (! -f $config_file_original) {
          copy($config_file, $config_file_original);
        }

        open my $in, "<", $config_file_original || die "Cannot open file lit.cfg.py.ori: $!";
        open my $out, ">", $config_file || die "Cannot open file lit.cfg.py: $!";
        while (<$in>) {
          s/env\s+SYCL_DEVICE_FILTER=(\S+)/env SYCL_DEVICE_FILTER=$1 $insert_command /g;
          print $out $_;
        }
        close $in;
        close $out;
    }

    sleep(30) if (is_windows()); # CMPLRTOOLS-26045: sleep for 30 second to correctly remove the folder
    rmtree($build_dir); # Clean build folder
    safe_Mkdir($build_dir); # Create build folder
    chdir_log($build_dir); # Enter build folder
    execute( "cmake -G Ninja ../ -DTEST_SUITE_SUBDIRS=$subdir -DTEST_SUITE_LIT=$lit"
                                          . " -DSYCL_BE=$sycl_backend -DSYCL_TARGET_DEVICES=$device"
                                          . " -DCMAKE_BUILD_TYPE=None" # to remove predifined options
                                          . " -DCMAKE_C_COMPILER=\"$c_cmplr\""
                                          . " -DCMAKE_CXX_COMPILER=\"$cpp_cmplr\""
                                          . " -DCMAKE_C_FLAGS=\"$c_cmd_opts\""
                                          . " -DCMAKE_CXX_FLAGS=\"$cpp_cmd_opts\""
                                          . " -DCMAKE_EXE_LINKER_FLAGS=\"$link_flags\""
                                          . " -DCMAKE_THREAD_LIBS_INIT=\"$thread_opts\""
                                          . " -DTEST_SUITE_COLLECT_CODE_SIZE=\"$collect_code_size\""
                                          . " -DLIT_EXTRA_ENVIRONMENT=\"$lit_extra_env\""
                                          . " $gpu_aot_target_opts"
                                          . " > $cmake_log 2>&1"
                                      );
    return $command_status, $command_output;
}

sub convert_opt
{
    my $opt = shift;

    # Convert options from MSVC format to clang format
    # For other options, keep them the original format
    $opt =~ s/[\/\-]Od/-O0/g;
    $opt =~ s/[\/]O([0-3]{1})/-O$1/g;
    $opt =~ s/\/Zi/-g/g;
    return $opt;
}

sub remove_opt
{
    my $cmplr_info = shift;

    my $cmplr = '';
    my $cmd_opts = '';
    if ( $cmplr_info =~ /([^\s]*)\s(.*)/)
    {
        $cmplr = $1;
        $cmd_opts = $2;
        chomp $cmd_opts;
        # Do not pass "-c" or "/c" arguments because some commands are executed with onestep
        $cmd_opts =~ s/[-\/]{1}c$|[-\/]{1}c\s{1,}//;
        # Do not pass "-fsycl" because it's included in the RUN commands
        $cmd_opts =~ s/-fsycl$|-fsycl\s{1,}//;
        # Do not pass "-fsycl-unnamed-lambda"
        $cmd_opts =~ s/-fsycl-unnamed-lambda$|-fsycl-unnamed-lambda\s{1,}//;
        # Remove "/EHsc" since it's not supported by clang/clang++
        $cmd_opts =~ s/\/EHsc$|\/EHsc\s{1,}//;
    } else {
        $cmplr = $cmplr_info;
    }
    return $cmplr, $cmd_opts;
}

sub report_result
{
    my $testname = shift;
    my $result = shift;
    my $message = shift;
    my $comp_output = shift;
    my $exec_output = shift;

    finalize_test($testname,
                  $result,
                  '', # status
                  0, # exesize
                  0, # objsize
                  0, # compile_time
                  0, # link_time
                  0, # execution_time
                  0, # save_time
                  0, # execute_time
                  $message,
                  0, # total_time
                  $comp_output,
                  $exec_output
    );
}

sub is_same
{
    my($array1, $array2) = @_;

    # Return 0 if two arrays are not the same length
    return 0 if scalar(@$array1) != scalar(@$array2);

    for(my $i = 0; $i <= $#$array1; $i++) {
        if ($array1->[$i] ne $array2->[$i]) {
           return 0;
        }
    }
    return 1;
}

sub join_extra_env
{
    my $extra_env = shift;
    my $env_var = shift;

    my $env = '';
    if (defined $ENV{$env_var}) {
        $env = "$env_var=$ENV{$env_var}";
        $extra_env = join(',',$extra_env,$env);
    }

    return $extra_env;
}

sub unxpath
{
    my $fpath;
    $fpath = shift;
    $fpath =~ s/\\/\//g;

    return $fpath;
}

sub file2str
{
    my $file = shift;
    ###
    local $/=undef;
    open FD, "<$file" or die "ERROR: Failed to open file $file!\n";
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
    open FD, ">$file" or die("ERROR: Failed to open $file for write.");

    print FD $s;
    close FD;
}

sub is_ats {
    my $current_gpu = $ENV{'CURRENT_GPU_DEVICE'};
    if (defined $current_gpu && $current_gpu =~ m/ats/i) {
      return 1;
    }
    return 0;
}

sub is_pvc {
    my $current_gpu = $ENV{'CURRENT_GPU_DEVICE'};
    if (defined $current_gpu && $current_gpu =~ m/pvc/i) {
      return 1;
    }
    return 0;
}

sub append2file
{
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">>$file" or die("ERROR: Failed to open $file for write.");

    my $last = '';
    while(<FD>) {
        if ($_ =~ /\Q$s\E/) {
            $last = $_;
            last;
        }
    }
    if ($last eq '') {
        print FD $s;
    }

    close FD;
}

sub CleanupTest {
  if ($current_test eq $test_to_run_list[-1]) {
    rename($run_all_lf, "$run_all_lf.last");
    rename($cmake_log, "$cmake_log.last");
  }
}

1;

