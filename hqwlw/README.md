# hqwlw

1111

## 性能检测数据可视化

项目新增脚本 `camera_check_fastapi/tools/visualize_performance_metrics.py`，用于将
`/metrics/performance` 接口返回的 JSON 生成可交互的 HTML 仪表盘。脚本仅依赖
Python 标准库，默认会从本地运行的 FastAPI 服务拉取数据并在当前目录生成
`performance_metrics.html`。

```bash
cd camera_check_fastapi/tools
python visualize_performance_metrics.py \
    http://localhost:8000/metrics/performance \
    --output performance.html
python -m webbrowser performance.html
```

如果已经导出了离线 JSON，可以将文件路径或 `-` (代表标准输入) 作为数据来源。

```bash
python visualize_performance_metrics.py snapshot.json --title "离线性能报告"
```

## MinIO 存储策略

- 系统始终以环形槽位（`safe_id/ring/{slot:06d}[ _intrussion].jpg`）覆写方式保存帧数据，不再额外生成时间戳文件，也无需手动修剪对象。
- 槽位指针通过 `MinioManager.next_ring_slot` 在进程内以异步锁维护，既避免了 Redis 依赖，又确保并发写入不会重复使用槽位。
- `MinioManager.initialize()` 会在桶创建后自动配置生命周期策略。生命周期天数优先读取 `config_settings.minio.lifecycle_days`，其次读取环境变量 `MINIO_LIFECYCLE_DAYS`，默认值为 3 天。

### 在其他模块中调用 MinioManager

```python
import asyncio
from camera_check_fastapi.src.main import MINIO_MANAGER, safe_filename, build_ring_obj_key

async def save_alarm_frame(stream_name: str, jpeg_bytes: bytes) -> None:
    if not MINIO_MANAGER.is_ready:
        return
    safe_id = safe_filename(stream_name)
    slot = await MINIO_MANAGER.next_ring_slot(safe_id, ring_size=120)
    if slot is None:
        return
    ring_key = build_ring_obj_key(safe_id, slot)
    await MINIO_MANAGER.put_bytes(ring_key, jpeg_bytes)

# 在 FastAPI 背景任务或其他协程上下文中调用
asyncio.create_task(save_alarm_frame("camera001", jpeg_bytes))
```

### MinioManager 使用示例

```python
from concurrent.futures import ThreadPoolExecutor
from camera_check_fastapi.src.main import MinioManager, build_ring_obj_key

manager = MinioManager(
    endpoint="127.0.0.1:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    bucket="frames",
    secure=False,
    lifecycle_days=7,  # 超过 7 天的对象将由 MinIO 自动清理
    io_executor=ThreadPoolExecutor(max_workers=32),
    upload_executor=ThreadPoolExecutor(max_workers=16),
)

manager.initialize()

async def write_frame(jpeg_bytes: bytes) -> None:
    slot = await manager.next_ring_slot("camera001", ring_size=120)
    if slot is None:
        return
    key = build_ring_obj_key("camera001", slot)
    await manager.put_bytes(key, jpeg_bytes)
```



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin http://192.168.130.162/zhangsihao/hqwlw.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](http://192.168.130.162/zhangsihao/hqwlw/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
