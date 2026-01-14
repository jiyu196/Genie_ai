# app/core/rule_filter.py
import csv
import re
import unicodedata
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("rule_filter")


class RuleFilter:
    """
    ✅ 규칙 기반 비속어 필터 (Java RuleFilter의 정확한 Python 포팅)
    - CSV 파일에서 forbidden → clean 매핑 로드
    - 문자열 정규화 (NFC, BOM 제거, zero-width 제거)
    - 정규식 기반 치환 (Java Matcher 방식과 동일하게)
    """

    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: CSV 파일 경로 (None이면 기본 경로 사용)
        """
        self.forbidden_map: Dict[str, str] = {}
        self.forbidden_pattern: Optional[re.Pattern] = None

        # CSV 경로 설정
        if csv_path is None:
            # app/model/forbidden_to_clean.csv 기본 경로
            csv_path = Path(__file__).parent.parent / "model" / "forbidden_to_clean.csv"

        self.csv_path = Path(csv_path)

        # 초기화
        self._load_csv()
        self._compile_pattern()

        logger.info(f"🚀 RuleFilter 초기화 완료 | 규칙 수: {len(self.forbidden_map)}")

    def _normalize(self, text: str) -> str:
        """
        문자열 정규화 (Java normalize 메서드와 완전히 동일)

        Java 코드:
        ```java
        Normalizer.normalize(s, Normalizer.Form.NFC)
            .replaceAll("\\uFEFF", "")   // BOM 제거
            .replaceAll("\\p{Cf}", "")   // zero-width 제거
            .trim();
        ```
        """
        if not text:
            return ""

        # 1. NFC 정규화 (자모 결합) - Java와 동일
        normalized = unicodedata.normalize('NFC', text)

        # 2. BOM 제거 (U+FEFF) - Java replaceAll("\\uFEFF", "") 와 동일
        normalized = normalized.replace('\uFEFF', '')

        # 3. zero-width 문자 제거 - Java replaceAll("\\p{Cf}", "") 와 동일
        # \p{Cf} = Format 카테고리 (zero-width, direction marks 등)
        normalized = ''.join(
            char for char in normalized
            if unicodedata.category(char) != 'Cf'
        )

        # 4. trim() - Java와 동일
        return normalized.strip()

    def _load_csv(self):
        """
        CSV 파일 로딩 (Java loadCsv 메서드와 완전히 동일)

        Java 로직:
        1. 헤더 스킵
        2. parts.length >= 3이면 index,forbidden,clean 형태
        3. parts.length >= 2이면 forbidden,clean 형태
        4. forbidden이 비어있지 않으면 맵에 추가
        5. clean이 비어있으면 공백(" ")으로 대체
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")

        try:
            # UTF-8 BOM 자동 제거 (encoding='utf-8-sig')
            with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)

                # Java: boolean firstLine = true 로직
                first_line = True

                for row in reader:
                    # Java: if (firstLine) { firstLine = false; continue; }
                    if first_line:
                        first_line = False
                        continue

                    # 빈 행 스킵
                    if not row:
                        continue

                    # Java: String[] parts = line.split(",", -1);
                    # CSV reader가 이미 파싱했으므로 row를 사용

                    # Java 로직과 정확히 동일
                    if len(row) >= 3:
                        # index,forbidden,clean 형태
                        forbidden = self._normalize(row[1])
                        clean = row[2].strip()
                    elif len(row) >= 2:
                        # forbidden,clean 형태
                        forbidden = self._normalize(row[0])
                        clean = row[1].strip()
                    else:
                        continue

                    # Java: if (!forbidden.isEmpty())
                    if forbidden:
                        # Java: clean.isEmpty() ? " " : clean
                        self.forbidden_map[forbidden] = " " if not clean else clean

            logger.info(f"✅ CSV 로딩 완료 | {len(self.forbidden_map)}개 규칙")

        except Exception as e:
            logger.error(f"❌ CSV 로딩 실패: {e}")
            raise RuntimeError(f"비속어 CSV 로딩 실패: {e}")

    def _compile_pattern(self):
        """
        정규식 패턴 컴파일 (Java compilePattern 메서드와 완전히 동일)

        Java 코드:
        ```java
        if (forbiddenMap.isEmpty()) {
            forbiddenPattern = Pattern.compile("(?!x)x");
            return;
        }

        String patternSource = forbiddenMap.keySet().stream()
                .sorted((a, b) -> Integer.compare(b.length(), a.length()))
                .map(Pattern::quote)
                .reduce((a, b) -> a + "|" + b)
                .orElse("");

        forbiddenPattern = Pattern.compile(patternSource);
        ```
        """
        # Java: if (forbiddenMap.isEmpty())
        if not self.forbidden_map:
            # Java: Pattern.compile("(?!x)x") - 매칭 불가능한 패턴
            self.forbidden_pattern = re.compile(r"(?!x)x")
            logger.warning("⚠️ forbidden_map이 비어있음")
            return

        # Java: sorted((a, b) -> Integer.compare(b.length(), a.length()))
        # 긴 단어부터 매칭하도록 정렬 (길이 내림차순)
        sorted_words = sorted(
            self.forbidden_map.keys(),
            key=len,
            reverse=True
        )

        # Java: .map(Pattern::quote)
        # 정규식 메타문자를 리터럴로 이스케이프
        escaped_words = [re.escape(word) for word in sorted_words]

        # Java: .reduce((a, b) -> a + "|" + b)
        # OR로 결합
        pattern_str = '|'.join(escaped_words)

        # Java: forbiddenPattern = Pattern.compile(patternSource);
        self.forbidden_pattern = re.compile(pattern_str)

        logger.info(f"✅ 정규식 패턴 컴파일 완료 | 패턴 길이: {len(pattern_str)}")

    def apply(self, text: str) -> str:
        """
        규칙 기반 치환 (Java apply 메서드와 완전히 동일)

        Java 코드:
        ```java
        String normalizedInput = normalize(input);

        Matcher matcher = forbiddenPattern.matcher(normalizedInput);
        StringBuffer sb = new StringBuffer(normalizedInput.length());
        while (matcher.find()) {
            String matched = matcher.group();
            String replacement = forbiddenMap.getOrDefault(matched, matched);
            matcher.appendReplacement(sb, Matcher.quoteReplacement(replacement));
        }
        matcher.appendTail(sb);
        return sb.toString();
        ```
        """
        # Java: if (input == null || input.isBlank())
        if not text or not text.strip():
            return text

        # Java: String normalizedInput = normalize(input);
        normalized_input = self._normalize(text)

        if not self.forbidden_pattern:
            logger.warning("⚠️ 패턴이 초기화되지 않음")
            return normalized_input

        # Java Matcher 방식과 동일하게 동작
        # matcher.find() -> matcher.group() -> matcher.appendReplacement()
        def replace_func(match):
            matched = match.group()
            # Java: forbiddenMap.getOrDefault(matched, matched)
            replacement = self.forbidden_map.get(matched, matched)
            return replacement

        # Java: while (matcher.find()) { ... matcher.appendReplacement() ... }
        # Java: matcher.appendTail(sb)
        result = self.forbidden_pattern.sub(replace_func, normalized_input)

        # 디버깅 로그 (Java 코드의 log.info와 대응)
        if logger.isEnabledFor(logging.DEBUG):
            if result != normalized_input:
                matches = list(self.forbidden_pattern.finditer(normalized_input))
                for m in matches:
                    logger.debug(f"✅ MATCHED: '{m.group()}' at position {m.start()}")
                logger.debug(f"🔍 Original: '{text}' | Normalized: '{normalized_input}'")

        return result


# 전역 싱글톤 인스턴스
_rule_filter_instance: Optional[RuleFilter] = None


def get_rule_filter() -> RuleFilter:
    """
    RuleFilter 싱글톤 인스턴스 반환

    Returns:
        RuleFilter 인스턴스
    """
    global _rule_filter_instance

    if _rule_filter_instance is None:
        try:
            _rule_filter_instance = RuleFilter()
        except Exception as e:
            logger.error(f"❌ RuleFilter 초기화 실패: {e}")
            raise

    return _rule_filter_instance


def apply_rule_filter(text: str) -> str:
    """
    편의 함수: 텍스트에 규칙 필터 적용

    Args:
        text: 원본 텍스트

    Returns:
        필터링된 텍스트
    """
    try:
        rule_filter = get_rule_filter()
        return rule_filter.apply(text)
    except Exception as e:
        logger.error(f"❌ 규칙 필터 적용 실패: {e}")
        # 실패 시 원본 반환 (안전장치)
        return text


def health_check() -> dict:
    """RuleFilter 상태 확인"""
    try:
        rule_filter = get_rule_filter()
        return {
            "initialized": True,
            "rules_count": len(rule_filter.forbidden_map),
            "csv_path": str(rule_filter.csv_path),
            "pattern_ready": rule_filter.forbidden_pattern is not None
        }
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e)
        }