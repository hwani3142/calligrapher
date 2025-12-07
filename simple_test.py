"""
간단한 테스트 스크립트 - TensorFlow 없이 기존 SVG 파일 확인
"""
import os

def check_environment():
    print("환경 구성 확인")
    print("-" * 50)

    # 체크포인트 파일 확인
    checkpoint_dir = 'checkpoints'
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        print(f"✓ 체크포인트 디렉토리: {len(files)}개 파일")
    else:
        print("✗ 체크포인트 디렉토리 없음")

    # 스타일 파일 확인
    styles_dir = 'styles'
    if os.path.exists(styles_dir):
        style_files = [f for f in os.listdir(styles_dir) if f.endswith('.npy')]
        print(f"✓ 스타일 파일: {len(style_files)}개")
    else:
        print("✗ 스타일 디렉토리 없음")

    # 기존 SVG 확인
    img_dir = 'img'
    if os.path.exists(img_dir):
        svg_files = [f for f in os.listdir(img_dir) if f.endswith('.svg')]
        print(f"✓ SVG 파일: {len(svg_files)}개")
        for svg in svg_files:
            size = os.path.getsize(os.path.join(img_dir, svg)) / 1024
            print(f"  - {svg}: {size:.1f} KB")
    else:
        print("✗ img 디렉토리 없음")

    print("\n" + "=" * 50)
    print("환경 구성 완료!")
    print("=" * 50)
    print("\n사용 방법:")
    print("1. Docker Desktop에서 Rosetta 2 활성화 (Settings > Features in development)")
    print("2. docker run --rm -v $(pwd)/img:/app/img calligrapher:tf1")
    print("\n또는:")
    print("conda activate calligrapher")
    print("python demo.py  # (TensorFlow 2.x 호환성 작업 필요)")

if __name__ == '__main__':
    check_environment()
