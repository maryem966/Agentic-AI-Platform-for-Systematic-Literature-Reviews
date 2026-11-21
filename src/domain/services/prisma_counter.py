"""
Service : PRISMACounter
Calcul des comptages PRISMA depuis les données brutes.
Section 6.2 du CdC.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PRISMAFlow:
    """
    Comptages PRISMA pour une revue systématique.
    Calculés depuis PostgreSQL — jamais stockés séparément.
    """

    # Identification
    identified_by_source: dict[str, int] = field(default_factory=dict)
    total_identified: int = 0

    # Déduplication
    duplicates_removed: int = 0
    after_deduplication: int = 0

    # Screening titre/résumé
    screened_title_abstract: int = 0
    excluded_title_abstract: int = 0
    excluded_reasons_title_abstract: dict[str, int] = field(default_factory=dict)

    # Screening texte intégral
    assessed_full_text: int = 0
    excluded_full_text: int = 0
    excluded_reasons_full_text: dict[str, int] = field(default_factory=dict)
    inaccessible: int = 0

    # Inclus
    included: int = 0

    def validate_consistency(self) -> list[str]:
        """
        Vérifie la cohérence des comptages PRISMA.
        Retourne la liste des erreurs (vide si cohérent).
        """
        errors: list[str] = []

        # Vérification 1 : somme par source == total_identified
        source_total = sum(self.identified_by_source.values())
        if self.identified_by_source and source_total != self.total_identified:
            errors.append(
                f"total_identified ({self.total_identified}) ≠ "
                f"somme par source ({source_total})"
            )

        # Vérification 2 : after_deduplication ≤ total_identified
        if self.after_deduplication > self.total_identified:
            errors.append("after_deduplication > total_identified — incohérence de déduplication")

        # Vérification 3 : assessed_full_text ≤ screened_title_abstract
        screened = self.screened_title_abstract or self.after_deduplication
        if self.assessed_full_text > screened:
            errors.append("assessed_full_text > screened — incohérence de screening")

        # Vérification 4 : included = assessed - excluded_ft - inaccessible
        if self.assessed_full_text > 0 and self.included > 0:
            expected = self.assessed_full_text - self.excluded_full_text - self.inaccessible
            if expected != self.included:
                errors.append(
                    f"included ({self.included}) ≠ "
                    f"assessed_ft - excluded_ft - inaccessible ({expected})"
                )

        return errors

    def to_prisma_text(self) -> str:
        """Génère le texte PRISMA pour le rapport."""
        lines = [
            f"Publications identifiées : {self.total_identified}",
            f"  Par source : {self.identified_by_source}",
            f"Après déduplication : {self.after_deduplication}",
            f"  Doublons supprimés : {self.duplicates_removed}",
            f"Screening titre/résumé : {self.screened_title_abstract}",
            f"  Exclus : {self.excluded_title_abstract}",
            f"Évalués texte intégral : {self.assessed_full_text}",
            f"  Exclus : {self.excluded_full_text}",
            f"  Inaccessibles : {self.inaccessible}",
            f"Inclus dans la revue : {self.included}",
        ]
        return "\n".join(lines)


class PRISMACounter:
    """
    Service de calcul des comptages PRISMA.

    En Phase 0 : implémentation stub.
    En Phase 2 : sera connecté aux repositories PostgreSQL réels.
    """

    def build_from_counts(
        self,
        identified_by_source: dict[str, int],
        duplicates_removed: int,
        excluded_title_abstract: int,
        excluded_title_abstract_reasons: dict[str, int],
        inaccessible: int,
        excluded_full_text: int,
        excluded_full_text_reasons: dict[str, int],
        included: int,
    ) -> PRISMAFlow:
        """Construit un PRISMAFlow depuis des comptages bruts."""
        total_identified = sum(identified_by_source.values())
        after_dedup = total_identified - duplicates_removed
        assessed_full_text = after_dedup - excluded_title_abstract
        screened = after_dedup

        return PRISMAFlow(
            identified_by_source=identified_by_source,
            total_identified=total_identified,
            duplicates_removed=duplicates_removed,
            after_deduplication=after_dedup,
            screened_title_abstract=screened,
            excluded_title_abstract=excluded_title_abstract,
            excluded_reasons_title_abstract=excluded_title_abstract_reasons,
            assessed_full_text=assessed_full_text,
            excluded_full_text=excluded_full_text,
            excluded_reasons_full_text=excluded_full_text_reasons,
            inaccessible=inaccessible,
            included=included,
        )
