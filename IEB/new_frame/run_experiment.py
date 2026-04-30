
import subprocess
import os
import sys
from datetime import datetime
import json
import time
#python train_student.py --ablation_mode baseline --teacher_path ../checkpoints/teacher/best_teacher.pt --data_dir ./pressure_vibration1_S2_02 --epochs 50 --batch_size 64 --lr 8e-4 --weight_decay 0.0001 --device cuda --seed 42 --num_workers 8 --save_dir checkpoints/ablation_full --auto_test --test_batch_size 64 --ieb_sdd_distill_weight 0

class AblationExperimentRunner:
    def __init__(self, log_dir='logs/experiments/ablation'):

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)


        self.experiments = []

    def add_experiment(self, name, command, description="", test_after_train=True):

        if test_after_train:
            if isinstance(command, str):
                if '--auto_test' not in command:
                    command += ' --auto_test'
            else:
                if '--auto_test' not in command:
                    command.append('--auto_test')

        self.experiments.append({
            'name': name,
            'command': command,
            'description': description,
            'test_after_train': test_after_train
        })

    def run_all(self, continue_on_error=True):

        total = len(self.experiments)
        results = []

        print(f"\n{'=' * 80}")
        print(f"开始运行 {total} 个消融实验")
        print(f"{'=' * 80}\n")

        for idx, exp in enumerate(self.experiments, 1):
            print(f"\n[{idx}/{total}] 实验: {exp['name']}")
            if exp['description']:
                print(f"描述: {exp['description']}")
            if exp.get('test_after_train', False):
                print(f"自动测试: 启用 ✓")
            print(f"命令: {exp['command']}")
            print(f"{'-' * 80}")

            # 运行实验
            success, duration, log_file = self._run_single_experiment(exp)

            # 记录结果
            result = {
                'name': exp['name'],
                'success': success,
                'duration': duration,
                'log_file': log_file,
                'test_after_train': exp.get('test_after_train', False),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            # 打印结果
            status = "✅ 成功" if success else "❌ 失败"
            print(f"\n{status} - 耗时: {duration:.2f}秒 ({duration / 60:.1f}分钟)")
            print(f"日志文件: {log_file}")


            if not success and not continue_on_error:
                print(f"\n实验失败，停止后续实验")
                break

            print(f"\n{'=' * 80}\n")

        self._save_summary(results)


        self._print_summary(results)

        return results

    def _run_single_experiment(self, exp):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f"{exp['name']}_{timestamp}.log")


        if isinstance(exp['command'], str):
            cmd = exp['command']
            use_shell = True
        else:
            cmd = exp['command']
            use_shell = False

        start_time = time.time()
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                # 写入实验信息
                f.write(f"实验名称: {exp['name']}\n")
                f.write(f"开始时间: {datetime.now().isoformat()}\n")
                f.write(f"命令: {cmd}\n")
                if exp.get('test_after_train', False):
                    f.write(f"自动测试: 启用\n")
                f.write(f"{'=' * 80}\n\n")
                f.flush()

                # 执行命令
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=use_shell,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )


                for line in process.stdout:
                    # 所有行都写入日志文件
                    f.write(line)
                    f.flush()

                    is_progress = ('it/s' in line or 's/it' in line) and '%' in line and '|' in line

                    if is_progress:

                        if '100%' in line:
                            print(line, end='')
                    else:

                        print(line, end='')

                process.wait()

                f.write(f"\n{'=' * 80}\n")
                f.write(f"结束时间: {datetime.now().isoformat()}\n")
                f.write(f"返回码: {process.returncode}\n")

                success = (process.returncode == 0)

        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n错误: {str(e)}\n")
            success = False

        duration = time.time() - start_time

        return success, duration, log_file

    def _save_summary(self, results):
        """保存实验总结"""
        summary_file = os.path.join(self.log_dir, 'ablation_summary.json')

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'with_auto_test': sum(1 for r in results if r.get('test_after_train', False)),
            'results': results
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n总结已保存到: {summary_file}")

    def _print_summary(self, results):
        """打印实验总结"""
        print(f"\n{'=' * 80}")
        print("消融实验总结")
        print(f"{'=' * 80}")

        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        with_test = sum(1 for r in results if r.get('test_after_train', False))

        print(f"\n总实验数: {total}")
        print(f"成功: {successful} ✅")
        print(f"失败: {failed} ❌")
        print(f"启用自动测试: {with_test}")
        print(f"\n详细结果:")

        for idx, result in enumerate(results, 1):
            status = "✅" if result['success'] else "❌"
            test_mark = " [含测试]" if result.get('test_after_train', False) else ""
            print(
                f"  {idx}. {status} {result['name']}{test_mark} ({result['duration']:.2f}秒 / {result['duration'] / 60:.1f}分钟)")
            print(f"     日志: {result['log_file']}")


def main():
    """主函数 - 配置并运行IEB-SDD消融实验"""
    runner = AblationExperimentRunner(log_dir='logs/experiments/ablation')

    # ========== 🆕 教师网络训练配置 ==========
    teacher_config = {
        'epochs': 30,
        'lr': 3e-4,  # 学习率 3e-4
        'weight_decay': 8e-4,  # 权重衰减 8e-4
        'dropout': 0.3,  # Dropout率 0.3
        'batch_size': 32,
        'device': 'cuda',
        'seed': 42,
        'num_workers': 0,
    }

    # ========== 通用配置 ==========
    # 基础参数（所有实验共享）
    base_config = {
        'epochs': 30,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'device': 'cuda',
        'seed': 42,
        'num_workers': 0,
        'eval_interval': 1,
        'log_interval': 10,
    }

    # 教师模型路径（必需）
    # 教师模型保存路径
    teacher_save_dir = 'checkpoints/teacher'
    teacher_path = './checkpoints/teacher/best_teacher.pt'

    # 数据路径
    data_dir = './pressure_vibration1_S2_02'
    runner.add_experiment(
        name='ablation_full',
        command=(
            f'python train_student.py '
            f'--ablation_mode full '
            f'--teacher_path {teacher_path} '
            f'--data_dir {data_dir} '
            f'--epochs 50 '
            f'--batch_size {base_config["batch_size"]} '
            f'--lr 1e-3 '
            f'--weight_decay {base_config["weight_decay"]} '
            f'--device {base_config["device"]} '
            f'--seed {base_config["seed"]} '
            f'--num_workers {base_config["num_workers"]} '
            f'--save_dir checkpoints/ablation_full '
            f'--test_modalities '
            f'--auto_test '
            #f'--use_pvnfm_inference '
            f'--test_batch_size 64 '
            f'--ieb_sdd_distill_weight 0.1 '
            f'--ieb_min_k 6 '
            f'--ieb_max_k 6 '
            f'--magnitude_decay_schedule piecewise '
            f'--magnitude_nfm_initial_weight 0.7 '
            f'--magnitude_nfm_final_weight 0.5 '
            f'--magnitude_nfm_warmup_ratio 0.3 '
            f'--magnitude_nfm_decay_ratio 0.5 '
            f'--pvnfm_inference_weight 0.8 '
        ),
        description='完整框架：IEB + SDD + PV-NFM',
        test_after_train=True
    )
    data_dir = './pressure_vibration1_S2_02'
    runner.add_experiment(
        name='ablation_full',
        command=(
            f'python train_student.py '
            f'--ablation_mode full '
            f'--teacher_path {teacher_path} '
            f'--data_dir {data_dir} '
            f'--epochs 50 '
            f'--batch_size {base_config["batch_size"]} '
            f'--lr 1e-3 '
            f'--weight_decay {base_config["weight_decay"]} '
            f'--device {base_config["device"]} '
            f'--seed {base_config["seed"]} '
            f'--num_workers {base_config["num_workers"]} '
            f'--save_dir checkpoints/ablation_full '
            f'--test_modalities '
            f'--auto_test '
            f'--use_pvnfm_inference '
            f'--test_batch_size 64 '
            f'--ieb_sdd_distill_weight 0.1 '
            f'--ieb_min_k 6 '
            f'--ieb_max_k 6 '
            f'--magnitude_decay_schedule piecewise '
            f'--magnitude_nfm_initial_weight 0.7 '
            f'--magnitude_nfm_final_weight 0.5 '
            f'--magnitude_nfm_warmup_ratio 0.3 '
            f'--magnitude_nfm_decay_ratio 0.5 '
            f'--pvnfm_inference_weight 0.8 '
        ),
        description='完整框架：IEB + SDD + PV-NFM',
        test_after_train=True
    )

    runner.add_experiment(
        name='ablation_full',
        command=(
            f'python train_student.py '
            f'--ablation_mode full '
            f'--teacher_path {teacher_path} '
            f'--data_dir {data_dir} '
            f'--epochs 50 '
            f'--batch_size {base_config["batch_size"]} '
            f'--lr 3e-3 '
            f'--weight_decay {base_config["weight_decay"]} '
            f'--device {base_config["device"]} '
            f'--seed {base_config["seed"]} '
            f'--num_workers {base_config["num_workers"]} '
            f'--save_dir checkpoints/ablation_full '
            f'--test_modalities '
            f'--auto_test '
            # f'--use_pvnfm_inference '
            f'--pvnfm_inference_weight 0.8 '
            f'--test_batch_size 64 '
            f'--ieb_sdd_distill_weight 0.1 '
            f'--ieb_min_k 6 '
            f'--ieb_max_k 6 '
            f'--magnitude_decay_schedule piecewise '
            f'--magnitude_nfm_initial_weight 0.7 '
            f'--magnitude_nfm_final_weight 0.5 '
            f'--magnitude_nfm_warmup_ratio 0.3 '
            f'--magnitude_nfm_decay_ratio 0.5 '
            f'--pvnfm_inference_weight 0.8 '
        ),
        description='完整框架：IEB + SDD + PV-NFM',
        test_after_train=True
    )
    data_dir = './pressure_vibration1_S2_02'
    runner.add_experiment(
        name='ablation_full',
        command=(
            f'python train_student.py '
            f'--ablation_mode full '
            f'--teacher_path {teacher_path} '
            f'--data_dir {data_dir} '
            f'--epochs 50 '
            f'--batch_size {base_config["batch_size"]} '
            f'--lr 3e-3 '
            f'--weight_decay {base_config["weight_decay"]} '
            f'--device {base_config["device"]} '
            f'--seed {base_config["seed"]} '
            f'--num_workers {base_config["num_workers"]} '
            f'--save_dir checkpoints/ablation_full '
            f'--test_modalities '
            f'--auto_test '
            f'--use_pvnfm_inference '
            f'--pvnfm_inference_weight 0.8 '
            f'--test_batch_size 64 '
            f'--ieb_sdd_distill_weight 0.1 '
            f'--ieb_min_k 6 '
            f'--ieb_max_k 6 '
            f'--magnitude_decay_schedule piecewise '
            f'--magnitude_nfm_initial_weight 0.7 '
            f'--magnitude_nfm_final_weight 0.5 '
            f'--magnitude_nfm_warmup_ratio 0.3 '
            f'--magnitude_nfm_decay_ratio 0.5 '
            f'--pvnfm_inference_weight 0.8 '
        ),
        description='完整框架：IEB + SDD + PV-NFM',
        test_after_train=True
    )




   # 1  前三个 都跑40epoch 都跑40epoch ieb_min_k 4 5 6 no_ieb
   #  runner.add_experiment(
   #      name='ablation_full',
   #      command=(
   #          f'python train_student.py '
   #          f'--ablation_mode full '
   #          f'--teacher_path {teacher_path} '
   #          f'--data_dir {data_dir} '
   #          f'--epochs 50 '
   #          f'--batch_size {base_config["batch_size"]} '
   #          f'--lr 3e-3 '
   #          f'--weight_decay {base_config["weight_decay"]} '
   #          f'--device {base_config["device"]} '
   #          f'--seed {base_config["seed"]} '
   #          f'--num_workers {base_config["num_workers"]} '
   #          f'--save_dir checkpoints/ablation_full '
   #          f'--auto_test '
   #          f'--test_batch_size 64 '
   #          f'--ieb_sdd_distill_weight 0.1 '
   #          f'--ieb_min_k 6 '
   #          f'--ieb_max_k 6 '
   #      ),
   #      description='完整框架：IEB + SDD + PV-NFM（预期最优性能）',
   #      test_after_train=True
   #  )


    # f'--auto_test '
            # f'--test_batch_size 64'
            # f'--eval_interval {base_config["eval_interval"]} '
            # f'--log_interval {base_config["log_interval"]} '

    print("\n" + "=" * 80)
    print("IEB-SDD消融实验配置")
    print("=" * 80)
    print(f"\n共配置 {len(runner.experiments)} 个实验:")
    for idx, exp in enumerate(runner.experiments, 1):
        print(f"  {idx}. {exp['name']}: {exp['description']}")

    print(f"\n基础配置:")
    for key, value in base_config.items():
        print(f"  - {key}: {value}")
    print(f"  - teacher_path: {teacher_path}")
    print(f"  - data_dir: {data_dir}")

    print(f"\n预期运行时间: ~{len(runner.experiments) * base_config['epochs'] * 2 / 60:.1f}小时")
    print(f"日志目录: {runner.log_dir}")

    # 确认运行
    print("\n" + "=" * 80)
    # response = input("是否开始运行所有实验？(y/n): ")
    # if response.lower() != 'y':
    #     print("已取消")
    #     return []

    # 运行实验
    results = runner.run_all(continue_on_error=True)

    return results
