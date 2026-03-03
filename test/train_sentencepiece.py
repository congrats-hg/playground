"""
SentencePiece BPE 모델 학습 스크립트
====================================
영어/한국어 SentencePiece 모델을 학습하고 저장합니다.

사용법:
    python train_sentencepiece.py

출력 파일:
    - en_train.txt, ko_train.txt  (학습용 텍스트)
    - en_spm.model, en_spm.vocab  (영어 SentencePiece 모델)
    - ko_spm.model, ko_spm.vocab  (한국어 SentencePiece 모델)
"""

import argparse
import time
import multiprocessing as mp
import sentencepiece as spm
from datasets import load_dataset


def _train_spm(input_file, model_prefix, vocab_size, character_coverage, max_sentence_length):
    """별도 프로세스에서 SentencePiece 모델을 학습한다."""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=character_coverage,
        max_sentence_length=max_sentence_length,
        shuffle_input_sentence=True,
    )


def main():
    parser = argparse.ArgumentParser(description="SentencePiece BPE 모델 학습")
    parser.add_argument("--num_samples", type=int, default=88000, help="학습에 사용할 문장 수")
    parser.add_argument("--vocab_size", type=int, default=16000, help="어휘 크기")
    args = parser.parse_args()

    num_samples = args.num_samples
    vocab_size = args.vocab_size

    total_start = time.time()

    # ============================================================
    # 1. 데이터 로드
    # ============================================================
    step_start = time.time()
    print("[1/3] 데이터 로드 중...")
    dataset = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task")
    elapsed = time.time() - step_start

    print(f"  전체 데이터: {len(dataset['train'])}개")
    print(f"  사용할 샘플: {num_samples}개")
    print(f"  소요 시간: {elapsed:.1f}초")

    # ============================================================
    # 2. 학습용 텍스트 파일 생성
    # ============================================================
    step_start = time.time()
    print("\n[2/3] 학습용 텍스트 파일 생성 중...")
    en_count = 0
    ko_count = 0

    with open("en_train.txt", "w", encoding="utf-8") as f_en, \
         open("ko_train.txt", "w", encoding="utf-8") as f_ko:
        for i, data in enumerate(dataset['train']):
            if i >= num_samples:
                break
            eng = data['english'].strip()
            kor = data['korean'].strip()
            f_en.write(eng + "\n")
            f_ko.write(kor + "\n")
            en_count += 1
            ko_count += 1

    elapsed = time.time() - step_start
    print(f"  영어 문장 수: {en_count}")
    print(f"  한국어 문장 수: {ko_count}")
    print(f"  소요 시간: {elapsed:.1f}초")

    # ============================================================
    # 3. SentencePiece 모델 학습 (영어/한국어 병렬)
    #    - BPE (Byte Pair Encoding) 방식
    #    - 특수 토큰: <pad>=0, <unk>=1, <sos>=2, <eos>=3
    #    - 서로 독립적이므로 별도 프로세스에서 동시 학습
    # ============================================================
    print(f"\n[3/3] SentencePiece 모델 병렬 학습 (vocab_size={vocab_size})")

    step_start = time.time()
    print("  영어 + 한국어 BPE 동시 학습 중...")

    en_proc = mp.Process(target=_train_spm, args=(
        'en_train.txt', 'en_spm', vocab_size, 1.0, 200,
    ))
    ko_proc = mp.Process(target=_train_spm, args=(
        'ko_train.txt', 'ko_spm', vocab_size, 0.9995, 300,
    ))

    en_proc.start()
    ko_proc.start()
    en_proc.join()
    ko_proc.join()

    parallel_elapsed = time.time() - step_start
    print(f"  병렬 학습 완료: {parallel_elapsed:.1f}초")

    # ============================================================
    # 4. 학습 결과 확인
    # ============================================================
    sp_en = spm.SentencePieceProcessor()
    sp_ko = spm.SentencePieceProcessor()
    sp_en.load('en_spm.model')
    sp_ko.load('ko_spm.model')

    print(f"\n{'=' * 50}")
    print(f"SentencePiece 모델 학습 완료!")
    print(f"{'=' * 50}")
    print(f"영어 어휘 크기: {sp_en.get_piece_size()}")
    print(f"한국어 어휘 크기: {sp_ko.get_piece_size()}")

    # 토큰화 테스트
    test_en = "Have you had dinner?"
    test_ko = "저녁은 드셨나요?"

    print(f"\n--- 토큰화 테스트 ---")
    print(f"영어: {test_en}")
    print(f"  토큰: {sp_en.encode(test_en, out_type=str)}")
    print(f"  정수: {sp_en.encode(test_en, out_type=int)}")

    print(f"한국어: {test_ko}")
    print(f"  토큰: {sp_ko.encode(test_ko, out_type=str)}")
    print(f"  정수: {sp_ko.encode(test_ko, out_type=int)}")

    # 특수 토큰 확인
    print(f"\n--- 특수 토큰 ---")
    print(f"PAD={sp_en.pad_id()}, UNK={sp_en.unk_id()}, BOS={sp_en.bos_id()}, EOS={sp_en.eos_id()}")

    total_elapsed = time.time() - total_start
    print(f"\n저장된 파일: en_spm.model, en_spm.vocab, ko_spm.model, ko_spm.vocab")
    print(f"총 소요 시간: {total_elapsed:.1f}초 ({total_elapsed / 60:.1f}분)")


if __name__ == "__main__":
    main()
