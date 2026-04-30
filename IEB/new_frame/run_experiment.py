"""
IEB-SDD消融实验批量运行脚本
基于NFM-IEB-SDD框架的5种消融实验自动化执行
"""

import subprocess
import os
import sys
from datetime import datetime
import json
import time
#python train_student.py --ablation_mode baseline --teacher_path ../checkpoints/teacher/best_teacher.pt --data_dir ./pressure_vibration1_S2_02 --epochs 50 --batch_size 64 --lr 8e-4 --weight_decay 0.0001 --device cuda --seed 42 --num_workers 8 --save_dir checkpoints/ablation_full --auto_test --test_batch_size 64 --ieb_sdd_distill_weight 0

class AblationExperimentRunner:
    """消融实验运行器"""

    def __init__(self, log_dir='logs/experiments/ablation'):
        """
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 实验配置列表
        self.experiments = []

    def add_experiment(self, name, command, description="", test_after_train=True):
        """
        添加一个实验

        Args:
            name: 实验名称（用于日志文件命名）
            command: 要执行的命令（字符串或列表）
            description: 实验描述
            test_after_train: 训练完成后是否自动测试（默认True）
        """
        # 如果启用自动测试，在命令中添加 --auto_test 参数
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
        """
        运行所有实验

        Args:
            continue_on_error: 如果某个实验失败，是否继续运行后续实验
        """
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

            # 如果失败且不继续，则停止
            if not success and not continue_on_error:
                print(f"\n实验失败，停止后续实验")
                break

            print(f"\n{'=' * 80}\n")

        # 保存总结
        self._save_summary(results)

        # 打印总结
        self._print_summary(results)

        return results

    def _run_single_experiment(self, exp):
        """
        运行单个实验

        Returns:
            success: 是否成功
            duration: 运行时长（秒）
            log_file: 日志文件路径
        """
        # 准备日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f"{exp['name']}_{timestamp}.log")

        # 准备命令
        if isinstance(exp['command'], str):
            cmd = exp['command']
            use_shell = True
        else:
            cmd = exp['command']
            use_shell = False

        # 运行命令
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

                # 实时输出并写入日志
                # 实时输出并写入日志（智能过滤进度条）
                # 实时输出并写入日志（智能过滤进度条）
                last_progress_prefix = None
                last_percent = -1  # 新增：记录上次输出的百分比
                # 实时输出并写入日志（智能过滤进度条）
                last_progress_key = None
                # 实时输出并写入日志（只保留100%进度条）
                for line in process.stdout:
                    # 所有行都写入日志文件
                    f.write(line)
                    f.flush()

                    # 检查是否是tqdm进度条行
                    is_progress = ('it/s' in line or 's/it' in line) and '%' in line and '|' in line

                    if is_progress:
                        # 只在屏幕上显示100%的进度条
                        if '100%' in line:
                            print(line, end='')
                    else:
                        # 非进度条行直接输出到屏幕（包括Train Loss等）
                        print(line, end='')

                # 等待完成
                process.wait()

                # 写入结束信息
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


if __name__ == '__main__':
    try:
        results = main()

        # 根据结果设置退出码
        if results:
            failed = sum(1 for r in results if not r['success'])
            sys.exit(0 if failed == 0 else 1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n用户中断实验")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
        # ========== 🆕 实验0: 训练教师网络 ==========
        # 1  中等学习率配置
        #  runner.add_experiment(
        #      name='train_teacher',
        #      command=(
        #          f'python train_teacher.py '
        #          f'--data_dir {data_dir} '
        #          f'--epochs 50 '
        #          f'--lr_cnn 5e-4 '
        #          f'--lr_projection 5e-4 '
        #          f'--lr_classifier 3e-4 '
        #
        #          f'--wd_cnn 1e-4 '
        #          f'--wd_projection 1e-4 '
        #          f'--wd_classifier 5e-4 '
        #          f'--dropout 0.3 '
        #
        #          f'--batch_size 32 '
        #          f'--device {teacher_config["device"]} '
        #          f'--seed {teacher_config["seed"]} '
        #          f'--num_workers {teacher_config["num_workers"]} '
        #          f'--save_dir {teacher_save_dir}'
        #      ),
        #
        #      description='训练IEB教师网络（FD-MVLLM架构）CNN编码器: lr=5e-4, 分类头: lr=3e-4',
        #      test_after_train=False  # 教师网络训练不需要自动测试
        #  )
